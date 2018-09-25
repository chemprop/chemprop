import math
import random
from typing import Callable, List, Tuple

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_auc_score
import torch.nn as nn


def get_data(path: str) -> List[Tuple[str, List[float]]]:
    """
    Gets smiles string and target values from a CSV file.

    :param path: Path to a CSV file.
    :return: A list of tuples where each tuple contains a smiles string and
    a list of target values (which are None if the target value is not specified).
    """
    data = []
    with open(path) as f:
        f.readline()  # remove header

        for line in f:
            line = line.strip().split(',')
            smiles = line[0]
            values = [float(x) if x != '' else None for x in line[1:]]
            data.append((smiles, values))

    return data


def split_data(data: List[Tuple[str, List[float]]],
               sizes: Tuple[float] = (0.8, 0.1, 0.1),
               seed: int = 0) -> Tuple[List[Tuple[str, List[float]]],
                                       List[Tuple[str, List[float]]],
                                       List[Tuple[str, List[float]]]]:
    """
    Splits data into training, validation, and test splits.

    :param data: A list of data points (smiles string, target values).
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3, sum(sizes) == 1

    random.seed(seed)
    random.shuffle(data)

    train_size, val_size = [int(size * len(data)) for size in sizes[:2]]

    train = data[:train_size]
    val = data[train_size:train_size + val_size]
    test = data[train_size + val_size:]

    return train, val, test


def get_loss_func(dataset_type: str) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param dataset_type: The dataset type ("classification" or "regression").
    :return: A PyTorch loss function.
    """
    if dataset_type == 'classification':
        return nn.BCELoss(reduction='none')

    if dataset_type == 'regression':
        return nn.MSELoss(reduction='none')

    raise ValueError('Dataset type "{}" not supported.'.format(dataset_type))


def get_metric_func(metric: str) -> Callable:
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: The name of the metric.
    :return: A metric function which takes as arguments a list of labels and a list of predictions.
    """
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        def metric_func(labels, preds):
            precision, recall, _ = precision_recall_curve(labels, preds)
            return auc(recall, precision)
        return metric_func

    if metric == 'rmse':
        return lambda labels, preds: math.sqrt(mean_squared_error(labels, preds))

    if metric == 'mae':
        return mean_absolute_error

    raise ValueError('Metric "{}" not supported.'.format(metric))
