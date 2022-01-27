from typing import Callable

import torch
import torch.nn as nn

from chemprop.args import TrainArgs


def get_loss_func(args: TrainArgs) -> Callable:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Arguments containing the dataset type ("classification", "regression", or "multiclass").
    :return: A PyTorch loss function.
    """

    # Nested dictionary of the form {dataset_type: {loss_function: loss_function callable}}, default keyed with None
    supported_loss_functions ={
        'regression':{
            None: nn.MSELoss(reduction='none'),
            'mse': nn.MSELoss(reduction='none'),
            'bounded_mse': bounded_mse
        },
        'classification':{
            None: nn.BCEWithLogitsLoss(reduction='none'),
            'cross_entropy': nn.BCEWithLogitsLoss(reduction='none'),
            'f1': f1_class_loss,
            'mcc': mcc_class_loss,
        },
        'multiclass':{
            None: nn.CrossEntropyLoss(reduction='none'),
            'cross_entropy': nn.CrossEntropyLoss(reduction='none'),
            'f1': f1_multiclass_loss,
            'mcc': mcc_multiclass_loss,
        },
        'spectra':{
            None: sid_loss,
            'spectra': sid_loss,
            'wasserstein': wasserstein_loss,
        }
    }

    # Error if no loss function supported
    if args.dataset_type not in supported_loss_functions.keys():
        raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

    # Return loss function if it is represented in the supported_loss_functions dictionary
    loss_function = supported_loss_functions.get(args.dataset_type, dict()).get(args.loss_function, None)

    if loss_function is not None:
        return loss_function

    elif args.loss_function is not None:
        raise ValueError(f'Loss function "{args.loss_function}" not supported with dataset type {args.dataset_type}. Available options are {supported_loss_functions[args.dataset_type].keys()}.')
    
    else: # loss_function is None
        raise ValueError(f'Default loss function not configured for dataset type {args.dataset_type}.')


def bounded_mse(predictions: torch.tensor, targets: torch.tensor, less_than_target: torch.tensor, greater_than_target: torch.tensor) -> torch.tensor:
    """
    
    """
    predictions = torch.where(
        torch.logical_and(predictions < targets, less_than_target),
        targets,
        predictions
    )

    predictions = torch.where(
        torch.logical_and(predictions > targets, greater_than_target),
        targets,
        predictions
    )
    
    return nn.functional.mseloss(predictions, targets, reduction='none')


def f1_class_loss(predictions: torch.tensor, targets: torch.tensor, data_weights: torch.tensor, mask: torch.tensor) -> torch.tensor:
    """
    
    """
    # shape(batch, tasks)
    # 2*TP/(2*TP + FN + FP)
    TP = torch.sum(targets * predictions * data_weights * mask, axis = 0)
    FP = torch.sum((1 - targets) * predictions * data_weights * mask, axis = 0)
    FN = torch.sum(targets * (1 - predictions) * data_weights * mask, axis = 0)
    return 2 * TP / (2 * TP + FN + FP)


def f1_multiclass_loss(predictions: torch.tensor, targets: torch.tensor, data_weights: torch.tensor, mask: torch.tensor) -> torch.tensor:
    """
    
    """
    # targets shape (batch)
    # preds shape(batch, classes)
    # 2*TP/(2*TP + FN + FP), FP = P - TP
    TP = torch.sum(predictions[torch.arange(targets.shape[0]), targets] * data_weights * mask)
    P = torch.sum(predictions * data_weights.unsqueeze(1) * mask.unsqueeze(1))
    FN = torch.sum(1 - predictions[torch.arange(targets.shape[0]), targets] * data_weights * mask)
    return 2 * TP / (TP + FN + P)


def mcc_class_loss(predictions: torch.tensor, targets: torch.tensor, data_weights: torch.tensor, mask: torch.tensor) -> torch.tensor:
    """
    
    """
    # shape(batch, tasks)
    # (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    TP = torch.sum(targets * predictions * data_weights * mask, axis = 0)
    FP = torch.sum((1 - targets) * predictions * data_weights * mask, axis = 0)
    FN = torch.sum(targets * (1 - predictions) * data_weights * mask, axis = 0)
    TN = torch.sum((1 - targets) * (1 - predictions) * data_weights * mask, axis = 0)
    return (TP*TN-FP*FN)/torch.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))


def mcc_multiclass_loss(predictions: torch.tensor, targets: torch.tensor, data_weights: torch.tensor, mask: torch.tensor) -> torch.tensor:
    """
    
    """
    # targets shape (batch)
    # preds shape(batch, classes)
    data_weights = data_weights.unsqueeze(1)
    mask = mask.unsqueeze(1)
    bin_targets = torch.zeros_like(predictions, device=predictions.device)
    bin_targets[torch.arange(predictions.shape[0]), targets] = 1
    c = torch.sum(predictions * bin_targets * data_weights * mask)
    s = torch.sum(predictions * data_weights * mask)
    pt = torch.sum(torch.sum(predictions * data_weights * mask, axis=0) * torch.sum(bin_targets * data_weights * mask, axis=0))
    p2 = torch.sum(torch.sum(predictions * data_weights * mask, axis=0)**2)
    t2 = torch.sum(torch.sum(bin_targets * data_weights * mask, axis=0)**2)
    return (c * s - pt) / torch.sqrt((s**2 - p2)*(s**2 - t2))


def sid_loss(model_spectra: torch.tensor, target_spectra: torch.tensor, mask: torch.tensor, threshold: float = None) -> torch.tensor:
    """
    Loss function for use with spectra data type.

    :param model_spectra: The predicted spectra output from a model with shape (batch_size,spectrum_length).
    :param target_spectra: The target spectra with shape (batch_size,spectrum_length). Values must be normalized so that each spectrum sums to 1.
    :param mask: Tensor with boolean indications of where the spectrum output should not be excluded with shape (batch_size,spectrum_length).
    :param threshold: Loss function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :return: A tensor containing loss values for the batch with shape (batch_size,spectrum_length).
    """
    # Move new tensors to torch device
    torch_device = model_spectra.device

    # Normalize the model spectra before comparison
    zero_sub = torch.zeros_like(model_spectra, device=torch_device)
    one_sub = torch.ones_like(model_spectra, device=torch_device)
    if threshold is not None:
        threshold_sub = torch.full(model_spectra.shape,threshold, device=torch_device)
        model_spectra = torch.where(model_spectra < threshold, threshold_sub, model_spectra)
    model_spectra = torch.where(mask, model_spectra, zero_sub)
    sum_model_spectra = torch.sum(model_spectra, axis=1, keepdim=True)
    model_spectra = torch.div(model_spectra, sum_model_spectra)

    # Calculate loss value
    target_spectra = torch.where(mask, target_spectra, one_sub)
    model_spectra = torch.where(mask, model_spectra, one_sub) # losses in excluded regions will be zero because log(1/1) = 0. 
    loss = torch.mul(torch.log(torch.div(model_spectra, target_spectra)), model_spectra) \
        + torch.mul(torch.log(torch.div(target_spectra, model_spectra)), target_spectra)
    loss = loss.to(torch_device)

    return loss


def wasserstein_loss(model_spectra: torch.tensor, target_spectra: torch.tensor, mask: torch.tensor, threshold: float = None) -> torch.tensor:
    """
    Loss function for use with spectra data type. This loss assumes that values are evenly spaced.

    :param model_spectra: The predicted spectra output from a model with shape (batch_size,spectrum_length).
    :param target_spectra: The target spectra with shape (batch_size,spectrum_length). Values must be normalized so that each spectrum sums to 1.
    :param mask: Tensor with boolian indications of where the spectrum output should not be excluded with shape (batch_size,spectrum_length).
    :param threshold: Loss function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :return: A tensor containing loss values for the batch with shape (batch_size,spectrum_length).
    """
    # Move new tensors to torch device
    torch_device = model_spectra.device

    # Normalize the model spectra before comparison
    zero_sub = torch.zeros_like(model_spectra, device=torch_device)
    if threshold is not None:
        threshold_sub = torch.full(model_spectra.shape,threshold, device=torch_device)
        model_spectra = torch.where(model_spectra < threshold, threshold_sub, model_spectra)
    model_spectra = torch.where(mask, model_spectra, zero_sub)
    sum_model_spectra = torch.sum(model_spectra, axis=1, keepdim=True)
    model_spectra = torch.div(model_spectra, sum_model_spectra)

    # Calculate loss value
    target_cum = torch.cumsum(target_spectra,axis=1).to(torch_device)
    model_cum = torch.cumsum(model_spectra,axis=1).to(torch_device)
    loss = torch.abs(target_cum - model_cum)
    loss = loss.to(torch_device)

    return loss
