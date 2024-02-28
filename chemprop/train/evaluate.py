from collections import defaultdict
import logging
from typing import Dict, List

import numpy as np

from .predict import predict
from chemprop.data import MoleculeDataLoader, StandardScaler, AtomBondScaler
from chemprop.models import MoleculeModel
from chemprop.train import get_metric_func


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metrics: List[str],
                         dataset_type: str,
                         is_atom_bond_targets: bool = False,
                         gt_targets: List[List[bool]] = None,
                         lt_targets: List[List[bool]] = None,
                         quantiles: List[float] = None,
                         logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates predictions using a metric function after filtering out invalid targets.

    :param preds: A list of lists of shape :code:`(data_size, num_tasks)` with model predictions.
    :param targets: A list of lists of shape :code:`(data_size, num_tasks)` with targets.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param is_atom_bond_targets: Boolean whether this is atomic/bond properties prediction.
    :param gt_targets: A list of lists of booleans indicating whether the target is an inequality rather than a single value.
    :param lt_targets: A list of lists of booleans indicating whether the target is an inequality rather than a single value.
    :param quantiles: A list of quantiles for use in pinball evaluation of quantile_interval metric.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    info = logger.info if logger is not None else print

    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    if len(preds) == 0:
        return {metric: [float('nan')] * num_tasks for metric in metrics}
    
    if is_atom_bond_targets:
        targets = [np.concatenate(x).reshape([-1, 1]) for x in zip(*targets)]

    # Filter out empty targets for most data types, excluding dataset_type spectra
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    valid_gt_targets = [[] for _ in range(num_tasks)]
    valid_lt_targets = [[] for _ in range(num_tasks)]
    if dataset_type != 'spectra':
        for i in range(num_tasks):
            if is_atom_bond_targets:
                for j in range(len(preds[i])):
                    if targets[i][j][0] is not None:  # Skip those without targets
                        valid_preds[i].append(list(preds[i][j]))
                        valid_targets[i].append(list(targets[i][j]))
                        if gt_targets is not None:
                            valid_gt_targets[i].append(list(gt_targets[i][j]))
                        if lt_targets is not None:
                            valid_lt_targets[i].append(list(lt_targets[i][j]))
            else:
                for j in range(len(preds)):
                    if targets[j][i] is not None:  # Skip those without targets
                        valid_preds[i].append(preds[j][i])
                        valid_targets[i].append(targets[j][i])
                        if gt_targets is not None:
                            valid_gt_targets[i].append(gt_targets[j][i])
                        if lt_targets is not None:
                            valid_lt_targets[i].append(lt_targets[j][i])

    # Compute metric. Spectra loss calculated for all tasks together, others calculated for tasks individually.
    results = defaultdict(list)
    if dataset_type == 'spectra':
        for metric, metric_func in metric_to_func.items():
            results[metric].append(metric_func(preds, targets))
    elif is_atom_bond_targets:
        for metric, metric_func in metric_to_func.items():
            if metric == 'quantile':
                if not quantiles:
                    raise ValueError("quantile metric evaluation requires quantiles parameter")
                for i, (valid_target, valid_pred) in enumerate(zip(valid_targets, valid_preds)):
                    valid_target = np.concatenate(valid_target)
                    valid_pred = np.concatenate(valid_pred)
                    results[metric].append(metric_func(valid_target, valid_pred, quantiles[i]))
            else:
                for valid_target, valid_pred in zip(valid_targets, valid_preds):
                    results[metric].append(metric_func(valid_target, valid_pred))
    else:
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
                if dataset_type == 'multiclass' and metric == 'cross_entropy':
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i],
                                                    labels=list(range(len(valid_preds[i][0])))))
                elif metric in ['bounded_rmse', 'bounded_mse', 'bounded_mae']:
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i], valid_gt_targets[i], valid_lt_targets[i]))
                elif metric == 'quantile':
                    if not quantiles:
                        raise ValueError("quantile metric evaluation requires quantiles parameter")
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i], quantiles[i]))
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
             quantiles: List[float] = None,
             atom_bond_scaler: AtomBondScaler = None,
             logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates an ensemble of models on a dataset by making predictions and then evaluating the predictions.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param quantiles: A list of quantiles for use in pinball evaluation of quantile_interval metric.
    :param atom_bond_scaler: A :class:`~chemprop.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.

    """
    # Inequality targets only need for evaluation of certain regression metrics
    if any(m in metrics for m in ['bounded_rmse', 'bounded_mse', 'bounded_mae']):
        gt_targets = data_loader.gt_targets
        lt_targets = data_loader.lt_targets
    else:
        gt_targets = None
        lt_targets = None

    preds = predict(
        model=model,
        data_loader=data_loader,
        scaler=scaler,
        atom_bond_scaler=atom_bond_scaler,
    )

    results = evaluate_predictions(
        preds=preds,
        targets=data_loader.targets,
        num_tasks=num_tasks,
        metrics=metrics,
        dataset_type=dataset_type,
        is_atom_bond_targets=model.is_atom_bond_targets,
        logger=logger,
        gt_targets=gt_targets,
        lt_targets=lt_targets,
        quantiles=quantiles,
    )

    return results
