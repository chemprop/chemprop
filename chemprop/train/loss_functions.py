from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from chemprop.args import TrainArgs


def get_loss_func(args: TrainArgs) -> Callable:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Arguments containing the dataset type ("classification", "regression", or "multiclass").
    :return: A PyTorch loss function.
    """

    supported_loss_functions = {
        "regression": {
            "mse": nn.MSELoss(reduction="none"),
            "bounded_mse": bounded_mse_loss,
            "mve": normal_mve,
            "evidential": evidential_loss,
        },
        "classification": {
            "binary_cross_entropy": nn.BCEWithLogitsLoss(reduction="none"),
            "mcc": mcc_class_loss,
            "dirichlet": dirichlet_class_loss,
        },
        "multiclass": {
            "cross_entropy": nn.CrossEntropyLoss(reduction="none"),
            "mcc": mcc_multiclass_loss,
            "dirichlet": dirichlet_multiclass_loss,
        },
        "spectra": {
            "sid": sid_loss,
            "wasserstein": wasserstein_loss,
        },
    }

    if args.dataset_type not in supported_loss_functions.keys():
        raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

    loss_function_choices = supported_loss_functions.get(args.dataset_type, dict())
    loss_function = loss_function_choices.get(args.loss_function)

    if loss_function is not None:
        return loss_function

    raise ValueError(
        f'Loss function: "{args.loss_function}" not supported for dataset type: '
        f' "{args.dataset_type}". Available options: {loss_function_choices.keys()}.'
    )


def bounded_mse_loss(
    predictions: Tensor,
    targets: Tensor,
    less_than_target: Tensor,
    greater_than_target: Tensor,
) -> Tensor:
    """
    Loss function for use with regression when some targets are presented as inequalities.

    :param predictions: Model predictions with shape(batch_size, tasks).
    :param targets: Target values with shape(batch_size, tasks).
    :param less_than_target: A tensor with boolean values indicating whether the target is a less-than inequality.
    :param greater_than_target: A tensor with boolean values indicating whether the target is a greater-than inequality.
    :return: A tensor containing loss values of shape(batch_size, tasks).
    """
    predictions = torch.where(
        torch.logical_and(predictions < targets, less_than_target), targets, predictions
    )

    predictions = torch.where(
        torch.logical_and(predictions > targets, greater_than_target),
        targets,
        predictions,
    )

    return F.mse_loss(predictions, targets, reduction="none")


def mcc_class_loss(
    predictions: Tensor,
    targets: Tensor,
    data_weights: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    A classification loss using a soft version of the Matthews Correlation Coefficient.

    :param predictions: a tensor of shape `b x t` containing model predictions, where `b` is the batch size and `t` ist the number of tasks
    :param targets: a tensor of shape `b x t` containing target values
    :param data_weights: A tensor of shape `b x 1` containing float values indicating how heavily to weight each datapoint in training with 
    :param mask: A tensor of shape `b x t` with boolean values indicating whether the loss for this prediction is considered in the gradient descent
    :return: A tensor of shape `b x t` containing loss values.
    """
    TP = (targets * predictions * data_weights * mask).sum(0)
    FP = ((1 - targets) * predictions * data_weights * mask).sum(0)
    FN = (targets * (1 - predictions) * data_weights * mask).sum(0)
    TN = ((1 - targets) * (1 - predictions) * data_weights * mask).sum(0)
    loss = 1 - ((TP * TN - FP * FN) / torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

    return loss


def mcc_multiclass_loss(
    predictions: Tensor,
    targets: Tensor,
    data_weights: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    A multiclass loss using a soft version of the Matthews Correlation Coefficient. Multiclass definition follows the version in sklearn documentation.

    :param predictions: a tensor of  shape `b x c` containing the model predictions.
    :param targets: a tensor of shape `b`, where `b` is the batch size containing the target values.
    :param data_weights: tensor of shape `b x 1` containing the weight of the respective training 
        datapoint
    :param mask: A tensor of shape `b` with boolean values indicating whether the loss for this 
        prediction is considered in the gradient descent.
    :return: A tensor value for the loss.
    """
    mask = mask.unsqueeze(1)
    bin_targets = torch.zeros_like(predictions, device=predictions.device)
    bin_targets[range(len(predictions)), targets] = 1

    c = torch.sum(predictions * bin_targets * data_weights * mask)
    s = torch.sum(predictions * data_weights * mask)
    pt = torch.sum(
        (predictions * data_weights * mask).sum(0) * (bin_targets * data_weights * mask).sum(0)
    )
    p2 = torch.sum((predictions * data_weights * mask).sum(0) ** 2)
    t2 = torch.sum(torch.sum(bin_targets * data_weights * mask).sum(0) ** 2)
    loss = 1 - (c * s - pt) / torch.sqrt((s ** 2 - p2) * (s ** 2 - t2))

    return loss


def sid_loss(
    model_spectra: Tensor,
    target_spectra: Tensor,
    mask: Tensor,
    threshold: float = None,
) -> Tensor:
    """
    Loss function for use with spectra data type.

    :param model_spectra: The predicted spectra output from a model with shape `b x l`, where `b`
        is the batch size and `l` is the spectrum length.
    :param target_spectra: The target spectra with shape `b x l`. Values must be normalized so that 
        each spectrum sums to 1.
    :param mask: Tensor with boolean indications of where the spectrum output should not be 
        excluded with shape `b x l`.
    :param threshold: Loss function requires that values are positive and nonzero. Values below the 
        threshold will be replaced with the threshold value.
    :return: A tensor containing loss values for the batch with shape `b x l`
    """
    device = model_spectra.device
    zero_sub = torch.zeros_like(model_spectra, device=device)
    one_sub = torch.ones_like(model_spectra, device=device)
    
    if threshold is not None:
        model_spectra = model_spectra.clamp(threshold, None)

    model_spectra = torch.where(mask, model_spectra, zero_sub)
    sum_model_spectra = torch.sum(model_spectra, axis=1, keepdim=True)
    model_spectra = torch.div(model_spectra, sum_model_spectra)

    target_spectra = torch.where(mask, target_spectra, one_sub)
    model_spectra = torch.where(mask, model_spectra, one_sub)

    return (
        torch.mul(torch.div(model_spectra, target_spectra).log(), model_spectra) 
        + torch.mul(torch.div(target_spectra, model_spectra).log(), target_spectra)
    )


def wasserstein_loss(
    model_spectra: Tensor,
    target_spectra: Tensor,
    mask: Tensor,
    threshold: float = None,
) -> Tensor:
    """
    Loss function for use with spectra data type. This loss assumes that values are evenly spaced.

    :param model_spectra: The predicted spectra output from a model with shape (batch_size,spectrum_length).
    :param target_spectra: The target spectra with shape (batch_size,spectrum_length). Values must be normalized so that each spectrum sums to 1.
    :param mask: Tensor with boolian indications of where the spectrum output should not be excluded with shape (batch_size,spectrum_length).
    :param threshold: Loss function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :return: A tensor containing loss values for the batch with shape (batch_size,spectrum_length).
    """
    device = model_spectra.device
    zero_sub = torch.zeros_like(model_spectra, device=device)

    if threshold is not None:
        model_spectra = torch.clamp(model_spectra, threshold, None)

    model_spectra = torch.where(mask, model_spectra, zero_sub)
    sum_model_spectra = torch.sum(model_spectra, axis=1, keepdim=True)
    model_spectra = torch.div(model_spectra, sum_model_spectra)

    target_cum = torch.cumsum(target_spectra, axis=1)
    model_cum = torch.cumsum(model_spectra, axis=1)
    loss = torch.abs(target_cum - model_cum)

    return loss


def normal_mve(pred_values, targets):
    """
    Use the negative log likelihood function of a normal distribution as a loss function used for 
    making simultaneous predictions of the mean and error distribution variance simultaneously.

    :param pred_values: Combined predictions of means and variances of shape(data, tasks*2).
        Means are first in dimension 1, followed by variances.
    :return: A tensor loss value.
    """
    pred_means, pred_var = torch.split(pred_values, pred_values.shape[1] // 2, dim=1)

    return torch.log(2 * torch.pi * pred_var) / 2 + (pred_means - targets) ** 2 / (2 * pred_var)


def dirichlet_class_loss(alphas, target_labels, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al in classification datasets.
    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, tasks*2).
                   Negative class first then positive class in dimension 1.
    :param target_labels: Digital labels to predict in shape(datapoints, tasks).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    num_tasks = target_labels.shape[1]
    num_classes = 2
    alphas = torch.reshape(alphas, (len(alphas), num_tasks, num_classes))

    y_one_hot = torch.eye(num_classes, device=alphas.device)[target_labels.long()]

    return dirichlet_common_loss(alphas=alphas, y_one_hot=y_one_hot, lam=lam)


def dirichlet_multiclass_loss(alphas, target_labels, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al for multiclass datasets.
    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, task, classes).
    :param target_labels: Digital labels to predict in shape(datapoints, tasks).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    num_classes = alphas.shape[2]

    y_one_hot = torch.eye(num_classes, device=alphas.device)[target_labels.long()]

    return dirichlet_common_loss(alphas=alphas, y_one_hot=y_one_hot, lam=lam)


def dirichlet_common_loss(alphas, y_one_hot, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al. This function follows
    after the classification and multiclass specific functions that reshape the 
    alpha inputs and create one-hot targets.

    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, task, classes).
    :param y_one_hot: Digital labels to predict in shape(datapoints, tasks, classes).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    S = torch.sum(alphas, dim=-1, keepdim=True)
    p = alphas / S
    A = torch.sum((y_one_hot - p) ** 2, dim=-1, keepdim=True)
    B = torch.sum((p * (1 - p)) / (S + 1), dim=-1, keepdim=True)
    L_sos = A + B

    alpha_hat = y_one_hot + (1 - y_one_hot) * alphas

    beta = torch.ones_like(alpha_hat)
    S_alpha = torch.sum(alpha_hat, dim=-1, keepdim=True)
    S_beta = torch.sum(beta, dim=-1, keepdim=True)

    ln_alpha = (
        torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha_hat), dim=-1, keepdim=True)
    )
    ln_beta = (
        torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)
    )

    dg_alpha = torch.digamma(alpha_hat)
    dg_S_alpha = torch.digamma(S_alpha)

    L_kl = (
        ln_alpha
        + ln_beta
        + torch.sum((alpha_hat - beta) * (dg_alpha - dg_S_alpha), dim=-1, keepdim=True)
    )
    L_kl = lam * L_kl

    return torch.mean(L_sos + L_kl, dim=-1)


# updated evidential regression loss (evidential_loss_new from Amini repo)
def evidential_loss(pred_values, targets, lam=0, epsilon=1e-8):
    """
    Use Deep Evidential Regression negative log likelihood loss and evidential regularizer

    :param pred_values: a tensor of shape `b x 4t`, where `b` is the batch size and `t` is the 
        number of tasks, containing prediction values for mu, v, alpha, and beta, respectively
    :param targets: a tensor of shape `b` containing the corresponding targets
    :param lam: the weight of the regularization term
    :param epsilon:

    :return: a tensor of shape `b` containing the evidential loss value for each input
    """
    mu, v, alpha, beta = torch.split(pred_values, pred_values.shape[1] // 4, dim=1)

    twoBlambda = 2 * beta * (1 + v)
    L_nll = (
        0.5 * torch.log(torch.pi / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    error = torch.abs((targets - mu))
    L_reg = error * (2 * v + alpha)

    return L_nll + lam * (L_reg - epsilon)
