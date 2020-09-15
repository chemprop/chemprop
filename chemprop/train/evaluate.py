from collections import defaultdict
import logging
from typing import Dict, List

from .predict import predict
from chemprop.data import MoleculeDataLoader, StandardScaler
from chemprop.models import MoleculeModel
from chemprop.utils import get_metric_func


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metrics: List[str],
                         dataset_type: str,
                         logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates predictions using a metric function after filtering out invalid targets.

    :param preds: A list of lists of shape :code:`(data_size, num_tasks)` with model predictions.
    :param targets: A list of lists of shape :code:`(data_size, num_tasks)` with targets.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    info = logger.info if logger is not None else print

    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    if len(preds) == 0:
        return {metric: [float('nan')] * num_tasks for metric in metrics}

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = defaultdict(list)
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
                for metric in metrics:
                    results[metric].append(float('nan'))
                continue

        if len(valid_targets[i]) == 0:
            continue

        for metric, metric_func in metric_to_func.items():
            if dataset_type == 'multiclass':
                results[metric].append(metric_func(valid_targets[i], valid_preds[i],
                                                   labels=list(range(len(valid_preds[i][0])))))
            else:
                results[metric].append(metric_func(valid_targets[i], valid_preds[i]))

    results = dict(results)

    return results


def evaluate(model: MoleculeModel,
             data_loader: MoleculeDataLoader,
             num_tasks: int,
             metrics: List[str],
             dataset_type: str,
             scaler: StandardScaler = None,
             logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates an ensemble of models on a dataset by making predictions and then evaluating the predictions.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.

    """
    preds = predict(
        model=model,
        data_loader=data_loader,
        scaler=scaler
    )

    results = evaluate_predictions(
        preds=preds,
        targets=data_loader.targets,
        num_tasks=num_tasks,
        metrics=metrics,
        dataset_type=dataset_type,
        logger=logger
    )

    return results
