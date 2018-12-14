from argparse import Namespace
from typing import Callable, Dict, List, Union

import torch.nn as nn
import numpy as  np

from .predict import predict
from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.utils import rmse


def evaluate_predictions(preds: Union[List[List[float]], Dict[str, List[List[float]]]],
                         targets: Union[List[List[float]], Dict[str, List[List[float]]]],
                         metric_func: Callable,
                         args: Namespace) -> Union[List[float], Dict[str, float]]:
    """
    Evaluates predictions using a metric function and filtering out invalid targets.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param targets: A list of lists of shape (data_size, num_tasks) with targets.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param args: Namespace
    :return: A list with the score for each task based on `metric_func`.
    """
    if args.dataset_type == 'unsupervised':
        num_tasks = 1
        data_size = len(preds)
        preds = [[p] for p in preds]

    elif args.dataset_type == 'bert_pretraining':
        num_tasks = 1
        data_size = len(preds['vocab'])
        features_targets = targets['features']
        targets = [[t] for t in targets['vocab']]
        features_preds = preds['features']
        preds = [[p] for p in preds['vocab']]

    else:
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
    if args.dataset_type == 'bert_pretraining':
        results = {
            'features': rmse(features_targets, features_preds) if features_targets is not None else None,
            'vocab': metric_func(valid_targets[0], valid_preds[0])
        }
    else:
        results = []
        for i in range(num_tasks):
            # # Skip if all targets are identical, otherwise we'll crash during classification
            if args.dataset_type == 'classification' and \
                    (all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i])):
                print('Warning: Found a task with all 0s or all 1s')
                results.append(0.5)  # just assume a baseline AUC of 0.5, which is just random guessing
                continue

            # TODO: handle this case better??? (this currently only happens when a feature is None for all molecules)
            if len(valid_targets[i]) == 0:
                continue
            results.append(metric_func(valid_targets[i], valid_preds[i]))

    if args.predict_features_and_task:
        # TODO: is this what we want results to be or do we want to split out the features somehow?
        results = results[:args.real_num_tasks] + [np.mean(results[args.real_num_tasks:])]

    return results


def evaluate(model: nn.Module,
             data: MoleculeDataset,
             metric_func: Callable,
             args: Namespace,
             scaler: StandardScaler = None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param args: Arguments.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list with the score for each task based on `metric_func`.
    """
    if args.maml:
        smiles, targets = [], []
        for task_idx in range(len(data.data[0].targets)):
            _, task_test_data = data.sample_maml_task(args, task_idx, seed=0)
            task_test_data = MoleculeDataset(task_test_data)
            smiles += task_test_data.smiles()
            targets += task_test_data.targets()
    else:
        smiles, targets = data.smiles(), data.targets()

    if args.dataset_type == 'bert_pretraining':
        # Only predict targets that are masked out
        targets['vocab'] = [target if mask == 0 else None for target, mask in zip(targets['vocab'], data.mask())]

    preds = predict(
        model=model,
        data=data,
        args=args,
        scaler=scaler,
        bert_save_memory=True
    )

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        metric_func=metric_func,
        args=args
    )

    return results
