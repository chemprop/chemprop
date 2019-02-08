from argparse import Namespace
import logging
from typing import Callable, Dict, List, Union

import torch.nn as nn

from .predict import predict
from chemprop.data import MoleculeDataset, StandardScaler


def evaluate_predictions(preds: Union[List[List[float]], Dict[str, List[List[float]]]],
                         targets: Union[List[List[float]], Dict[str, List[List[float]]]],
                         metric_func: Callable,
                         dataset_type: str,
                         args: Namespace = None,
                         logger: logging.Logger = None) -> Union[List[float], Dict[str, float]]:
    """
    Evaluates predictions using a metric function and filtering out invalid targets.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param targets: A list of lists of shape (data_size, num_tasks) with targets.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param dataset_type: Dataset type.
    :param args: Namespace
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    info = logger.info if logger is not None else print
    data_size, num_tasks = len(preds), len(preds[0])
    
    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(data_size):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = []
    for i in range(num_tasks):
        # # Skip if all targets or preds are identical, otherwise we'll crash during classification
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                info('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                info('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                results.append(float('nan'))
                continue

        if len(valid_targets[i]) == 0:
            continue
        results.append(metric_func(valid_targets[i], valid_preds[i]))

    return results


def evaluate(model: nn.Module,
             data: MoleculeDataset,
             metric_func: Callable,
             args: Namespace,
             scaler: StandardScaler = None,
             logger: logging.Logger = None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param dataset_type: Dataset type.
    :param args: Arguments.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    preds = predict(
        model=model,
        data=data,
        batch_size=args.batch_size,
        scaler=scaler
    )

    targets = data.targets()

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        args=args,
        logger=logger
    )

    return results
