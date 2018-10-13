import logging
import math
import os
import random
from copy import deepcopy
from typing import Callable, List, Tuple, Union

import numpy as np
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score, roc_auc_score
import torch.nn as nn


class StandardScaler:
    def fit(self, X: List[List[float]]) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0-th axis.

        :param X: A list of lists of floats.
        :return: The fitted StandardScaler.
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)

        return self

    def transform(self, X: List[List[float]]):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats.
        :return: The transformed data.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means)/self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), None, transformed_with_nan)

        return transformed_with_none
    
    def inverse_transform(self, X: List[List[float]]):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), None, transformed_with_nan)

        return transformed_with_none


def convert_to_classes(data: List[Tuple[str, List[float]]], num_bins: int = 20) -> Tuple[List[Tuple[str, List[float]]],
                                                                                         np.ndarray,
                                                                                         List[Tuple[str, List[float]]]]:
    """
    Converts regression data to classification data by binning.

    :param data: Regression data in the format (smiles, [value_1, value_2, ...]).
    :param num_bins: The number of bins to use when doing regression_with_binning.
    :return: A tuple with the new classification data, a numpy array with the bin centers,
    and the original regression data.
    """
    print('Num bins for binning: {}'.format(num_bins))
    old_data = deepcopy(data)
    num_tasks = len(data[0][1])
    for task in range(num_tasks):
        regress = np.array([d[1][task] for d in data])
        bin_edges = np.quantile(regress, [float(i)/float(num_bins) for i in range(num_bins+1)])

        for i in range(len(data)):
            bin_index = (bin_edges <= regress[i]).sum() - 1
            bin_index = min(bin_index, num_bins-1)
            data[i][1][task] = bin_index

    return data, np.array([(bin_edges[i] + bin_edges[i+1])/2 for i in range(num_bins)]), old_data


def get_task_names(path: str, use_compound_names: bool = False) -> List[str]:
    """
    Gets the task names from a data CSV file.

    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A list of task names.
    """
    index = 2 if use_compound_names else 1
    with open(path) as f:
        task_names = f.readline().strip().split(',')[index:]

    return task_names


def get_data(path: str,
             dataset_type: str = None,
             num_bins: int = 20,
             use_compound_names: bool = False,
             get_header: bool = False) -> Union[Tuple[List[str], List[Tuple[str, List[float]]]],
                                                List[str], Tuple[List[str], List[Tuple[str, List[float]]]],
                                                List[str], List[str], Tuple[List[str], List[Tuple[str, List[float]]]]]:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    :param path: Path to a CSV file.
    :param dataset_type: The type of the dataset (ex. classification, regression, regression_with_binning).
    :param num_bins: The number of bins to use when doing regression_with_binning.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :param get_header: Whether to get the header in addition to the data.
    :return: A tuple where the first element is a list containing the header strings
     and the second element is a list of tuples where each tuple contains a smiles string and
    a list of target values (which are None if the target value is not specified).
    """
    data = []
    if use_compound_names:
        compound_names = []

    with open(path) as f:
        header = f.readline().strip().split(',')

        for line in f:
            line = line.strip().split(',')
            if use_compound_names:
                names = line[0]
                smiles = line[1]
                values = [float(x) if x != '' else None for x in line[2:]]
                compound_names.append(names)
            else:
                smiles = line[0]
                values = [float(x) if x != '' else None for x in line[1:]]
            data.append((smiles, values))

    if dataset_type == 'regression_with_binning':
        data = convert_to_classes(data, num_bins)

    if use_compound_names and get_header:
        return header, compound_names, data

    if use_compound_names:
        return compound_names, data

    if get_header:
        return header, data

    return data


def split_data(data: List[Tuple[str, List[float]]],
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
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


def truncate_outliers(data: List[Tuple[str, List[float]]]) -> List[Tuple[str, List[float]]]:
    """Truncates outlier values in a regression dataset.

    Every value which is outside mean ± 3 * std are truncated to equal mean ± 3 * std.

    :param data: A list of data points (smiles string, target values).
    :return: The same data but with outliers truncated.
    """
    # Determine mean and standard deviation by task
    smiles, values = zip(*data)
    values_by_task = np.array(values).T
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0)

    # Truncate values
    for i, task_values in enumerate(values_by_task):
        values_by_task[i] = np.clip(task_values, means[i] - 3 * stds[i], means[i] + 3 * stds[i])

    # Reconstruct data
    values = values_by_task.T.tolist()
    data = list(zip(smiles, values))

    return data


def get_loss_func(dataset_type: str) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param dataset_type: The dataset type ("classification" or "regression" or "regression_with_binning").
    :return: A PyTorch loss function.
    """
    if dataset_type == 'classification':
        return nn.BCELoss(reduction='none')
    
    if dataset_type == 'regression_with_binning':
        return nn.CrossEntropyLoss(reduction='none')

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

    if metric == 'r2':
        return r2_score

    raise ValueError('Metric "{}" not supported.'.format(metric))


def set_logger(logger: logging.Logger, save_dir: str, quiet: bool):
    """
    Sets up a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param logger: A logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    """
    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)

    fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
    fh_v.setLevel(logging.DEBUG)
    fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
    fh_q.setLevel(logging.INFO)

    logger.addHandler(ch)
    logger.addHandler(fh_v)
    logger.addHandler(fh_q)


def compute_pnorm(model: nn.Module) -> float:
    """Computes the norm of the parameters of a model."""
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """Computes the norm of the gradients of a model."""
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters()]))
