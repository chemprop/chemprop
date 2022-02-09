from typing import Callable

import torch
import torch.nn as nn
import numpy as np

from chemprop.args import TrainArgs


def get_loss_func(args: TrainArgs) -> Callable:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Arguments containing the dataset type ("classification", "regression", or "multiclass").
    :return: A PyTorch loss function.
    """

    # Nested dictionary of the form {dataset_type: {loss_function: loss_function callable}}
    supported_loss_functions ={
        'regression':{
            'mse': nn.MSELoss(reduction='none'),
            'bounded_mse': bounded_mse_loss
        },
        'classification':{
            'binary_cross_entropy': nn.BCEWithLogitsLoss(reduction='none'),
            'mcc': mcc_class_loss,
        },
        'multiclass':{
            'cross_entropy': nn.CrossEntropyLoss(reduction='none'),
            'mcc': mcc_multiclass_loss,
        },
        'spectra':{
            'sid': sid_loss,
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

    else:
        raise ValueError(f'Loss function "{args.loss_function}" not supported with dataset type {args.dataset_type}. \
            Available options for that dataset type are {supported_loss_functions[args.dataset_type].keys()}.')


def bounded_mse_loss(predictions: torch.tensor, targets: torch.tensor, less_than_target: torch.tensor, greater_than_target: torch.tensor) -> torch.tensor:
    """
    Loss function for use with regression when some targets are presented as inequalities.

    :param predictions: Model predictions with shape(batch_size, tasks).
    :param targets: Target values with shape(batch_size, tasks).
    :param less_than_target: A tensor with boolean values indicating whether the target is a less-than inequality.
    :param greater_than_target: A tensor with boolean values indicating whether the target is a greater-than inequality.
    :return: A tensor containing loss values of shape(batch_size, tasks).
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
    
    return nn.functional.mse_loss(predictions, targets, reduction='none')


def mcc_class_loss(predictions: torch.tensor, targets: torch.tensor, data_weights: torch.tensor, mask: torch.tensor) -> torch.tensor:
    """
    A classification loss using a soft version of the Matthews Correlation Coefficient.

    :param predictions: Model predictions with shape(batch_size, tasks).
    :param targets: Target values with shape(batch_size, tasks).
    :param data_weights: A tensor with float values indicating how heavily to weight each datapoint in training with shape(batch_size, 1)
    :param mask: A tensor with boolean values indicating whether the loss for this prediction is considered in the gradient descent with shape(batch_size, tasks).
    :return: A tensor containing loss values of shape(tasks).
    """
    # shape(batch, tasks)
    # (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    torch_device = predictions.device
    TP = torch.sum(targets * predictions * data_weights * mask, axis = 0).to(torch_device)
    FP = torch.sum((1 - targets) * predictions * data_weights * mask, axis = 0).to(torch_device)
    FN = torch.sum(targets * (1 - predictions) * data_weights * mask, axis = 0).to(torch_device)
    TN = torch.sum((1 - targets) * (1 - predictions) * data_weights * mask, axis = 0).to(torch_device)
    loss = 1 - ((TP*TN-FP*FN)/torch.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    loss = loss.to(torch_device)
    return loss


def mcc_multiclass_loss(predictions: torch.tensor, targets: torch.tensor, data_weights: torch.tensor, mask: torch.tensor) -> torch.tensor:
    """
    A multiclass loss using a soft version of the Matthews Correlation Coefficient. Multiclass definition follows the version in sklearn documentation.

    :param predictions: Model predictions with shape(batch_size, classes).
    :param targets: Target values with shape(batch_size).
    :param data_weights: A tensor with float values indicating how heavily to weight each datapoint in training with shape(batch_size, 1)
    :param mask: A tensor with boolean values indicating whether the loss for this prediction is considered in the gradient descent with shape(batch_size).
    :return: A tensor value for the loss.
    """
    # targets shape (batch)
    # preds shape(batch, classes)
    torch_device = predictions.device
    mask = mask.unsqueeze(1)
    bin_targets = torch.zeros_like(predictions, device=torch_device)
    bin_targets[torch.arange(predictions.shape[0]), targets] = 1
    c = torch.sum(predictions * bin_targets * data_weights * mask).to(torch_device)
    s = torch.sum(predictions * data_weights * mask).to(torch_device)
    pt = torch.sum(torch.sum(predictions * data_weights * mask, axis=0) * torch.sum(bin_targets * data_weights * mask, axis=0)).to(torch_device)
    p2 = torch.sum(torch.sum(predictions * data_weights * mask, axis=0)**2).to(torch_device)
    t2 = torch.sum(torch.sum(bin_targets * data_weights * mask, axis=0)**2).to(torch_device)
    loss = 1 - (c * s - pt) / torch.sqrt((s**2 - p2)*(s**2 - t2))
    loss = loss.to(torch_device)
    return loss


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
        threshold_sub = torch.full(model_spectra.shape, threshold, device=torch_device)
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


def negative_log_likelihood(pred_targets, pred_var, targets):
    clamped_var = torch.clamp(pred_var, min=0.00001)
    return torch.log(2*np.pi*clamped_var) / 2 + (pred_targets - targets)**2 / (2 * clamped_var)


# evidential classification
def dirichlet_loss(y, alphas, lam=1):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al
    :y: labels to predict
    :alphas: predicted parameters for Dirichlet
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    def KL(alpha):
        """
        Compute KL for Dirichlet defined by alpha to uniform dirichlet
        :alpha: parameters for Dirichlet

        :return: KL
        """
        beta = torch.ones_like(alpha)
        S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        S_beta = torch.sum(beta, dim=-1, keepdim=True)

        ln_alpha = torch.lgamma(S_alpha)-torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
        ln_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)

        # digamma terms
        dg_alpha = torch.digamma(alpha)
        dg_S_alpha = torch.digamma(S_alpha)

        # KL
        kl = ln_alpha + ln_beta + torch.sum((alpha - beta)*(dg_alpha - dg_S_alpha), dim=-1, keepdim=True)
        return kl


    # Hard code to 2 classes per task, since this assumption is already made
    # for the existing chemprop classification tasks
    num_classes = 2
    num_tasks = y.shape[1]

    y_one_hot = torch.eye(num_classes)[y.long()]
    if y.is_cuda:
        y_one_hot = y_one_hot.cuda()

    alphas = torch.reshape(alphas, (alphas.shape[0], num_tasks, num_classes))

    # SOS term
    S = torch.sum(alphas, dim=-1, keepdim=True)
    p = alphas / S
    A = torch.sum(torch.pow((y_one_hot - p), 2), dim=-1, keepdim=True)
    B = torch.sum((p*(1 - p)) / (S+1), dim=-1, keepdim=True)
    SOS = A + B

    # KL
    alpha_hat = y_one_hot + (1-y_one_hot)*alphas
    KL = lam * KL(alpha_hat)

    #loss = torch.mean(SOS + KL)
    loss = SOS + KL
    loss = torch.mean(loss, dim=-1)
    return loss


# updated evidential regression loss (evidential_loss_new from Amini repo)
def evidential_loss(mu, v, alpha, beta, targets, lam=1, epsilon=1e-4):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict

    :return: Loss
    """
    # Calculate NLL loss
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) \
        - alpha*torch.log(twoBlambda) \
        + (alpha+0.5) * torch.log(v*(targets-mu)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha+0.5)

    L_NLL = nll #torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg #torch.mean(reg, dim=-1)

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return loss