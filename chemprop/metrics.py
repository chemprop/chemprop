from typing import List, Callable, Union

from tqdm import trange
import torch
import numpy as np
import torch.nn as nn

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss, f1_score, matthews_corrcoef


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    r"""
    Gets the metric function corresponding to a given metric name.

    Supports:

    * :code:`auc`: Area under the receiver operating characteristic curve
    * :code:`prc-auc`: Area under the precision recall curve
    * :code:`rmse`: Root mean squared error
    * :code:`mse`: Mean squared error
    * :code:`mae`: Mean absolute error
    * :code:`r2`: Coefficient of determination R\ :superscript:`2`
    * :code:`accuracy`: Accuracy (using a threshold to binarize predictions)
    * :code:`cross_entropy`: Cross entropy
    * :code:`binary_cross_entropy`: Binary cross entropy
    * :code:`sid`: Spectral information divergence
    * :code:`wasserstein`: Wasserstein loss for spectra

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mse':
        return mean_squared_error

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'accuracy':
        return accuracy

    if metric == 'cross_entropy':
        return log_loss
    
    if metric == 'f1':
        return f1

    if metric == 'mcc':
        return mcc

    if metric == 'binary_cross_entropy':
        return bce
    
    if metric == 'sid':
        return sid_metric
    
    if metric == 'wasserstein':
        return wasserstein_metric

    raise ValueError(f'Metric "{metric}" not supported.')


def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def bce(targets: List[int], preds: List[float]) -> float:
    """
    Computes the binary cross entropy loss.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed binary cross entropy.
    """
    # Don't use logits because the sigmoid is added in all places except training itself
    bce_func = nn.BCELoss(reduction='mean')
    loss = bce_func(target=torch.Tensor(targets), input=torch.Tensor(preds)).item()

    return loss


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return mean_squared_error(targets, preds, squared=False)


def accuracy(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed accuracy.
    """
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction

    return accuracy_score(targets, hard_preds)


def f1(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    Computes the f1 score of a binary prediction task using a given threshold for generating hard predictions.

    Will calculate for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed f1 score.
    """
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
        score = f1_score(targets, hard_preds, average='micro')
    else: # binary prediction
        hard_preds = [1 if p > threshold else 0 for p in preds]  
        score = f1_score(targets, hard_preds)

    return score


def mcc(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    Computes the Matthews Correlation Coefficient of a binary prediction task using a given threshold for generating hard predictions.

    Will calculate for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed accuracy.
    """
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction

    return matthews_corrcoef(targets, hard_preds)


def sid_metric(model_spectra: List[List[float]], target_spectra: List[List[float]], threshold: float = None, batch_size: int = 50) -> float:
    """
    Metric function for use with spectra data type.

    :param model_spectra: The predicted spectra output from a model with shape (num_data, spectrum_length).
    :param target_spectra: The target spectra with shape (num_data, spectrum_length). Values must be normalized so that each spectrum sums to 1.
        Excluded values in target spectra will have a value of None.
    :param threshold: Function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :param batch_size: Batch size for calculating metric.
    :return: The average SID value for the predicted spectra.
    """
    losses = []
    num_iters, iter_step = len(model_spectra), batch_size

    for i in trange(0, num_iters, iter_step):

        # Create batches
        batch_preds = model_spectra[i:i + iter_step]
        batch_preds = np.array(batch_preds)
        batch_targets = target_spectra[i:i + iter_step]
        batch_mask = np.array([[x is not None for x in b] for b in batch_targets])
        batch_targets = np.array([[1 if x is None else x for x in b] for b in batch_targets])

        # Normalize the model spectra before comparison
        if threshold is not None:
            batch_preds[batch_preds < threshold] = threshold
        batch_preds[~batch_mask] = 0
        sum_preds = np.sum(batch_preds, axis=1, keepdims=True)
        batch_preds = batch_preds / sum_preds

        # Calculate loss value
        batch_preds[~batch_mask] = 1 # losses in excluded regions will be zero because log(1/1) = 0.
        loss = batch_preds * np.log(batch_preds / batch_targets) + batch_targets * np.log(batch_targets / batch_preds)
        loss = np.sum(loss, axis=1)

        # Gather batches
        loss = loss.tolist()
        losses.extend(loss)

    loss = np.mean(loss)

    return loss


def wasserstein_metric(model_spectra: List[List[float]], target_spectra: List[List[float]], threshold: float = None, batch_size: int = 50) -> float:
    """
    Metric function for use with spectra data type. This metric assumes that values are evenly spaced.

    :param model_spectra: The predicted spectra output from a model with shape (num_data, spectrum_length).
    :param target_spectra: The target spectra with shape (num_data, spectrum_length). Values must be normalized so that each spectrum sums to 1.
        Excluded values in target spectra will have value None.
    :param threshold: Function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :param batch_size: Batch size for calculating metric.
    :return: The average wasserstein loss value for the predicted spectra.
    """
    losses = []
    num_iters, iter_step = len(model_spectra), batch_size

    for i in trange(0, num_iters, iter_step):

        # Create batches
        batch_preds = model_spectra[i:i + iter_step]
        batch_preds = np.array(batch_preds)
        batch_targets = target_spectra[i:i + iter_step]
        batch_mask = np.array([[x is not None for x in b] for b in batch_targets])
        batch_targets = np.array([[0 if x is None else x for x in b] for b in batch_targets])

        # Normalize the model spectra before comparison
        if threshold is not None:
            batch_preds[batch_preds < threshold] = threshold
        batch_preds[~batch_mask] = 0
        sum_preds = np.sum(batch_preds, axis=1, keepdims=True)
        batch_preds = batch_preds / sum_preds

        # Calculate loss value
        target_cum = np.cumsum(batch_targets,axis=1)
        preds_cum = np.cumsum(batch_preds,axis=1)
        loss = np.abs(target_cum - preds_cum)
        loss = np.sum(loss, axis=1)

        # Gather batches
        loss = loss.tolist()
        losses.extend(loss)

    loss = np.mean(loss)

    return loss
