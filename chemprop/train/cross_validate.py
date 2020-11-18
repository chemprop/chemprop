from collections import defaultdict
import csv
from logging import Logger
import os
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from .run_training import run_training
from chemprop.args import TrainArgs
from chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data import get_data, get_task_names, MoleculeDataset, validate_dataset_type
from chemprop.utils import create_logger, makedirs, timeit
from chemprop.features import set_extra_atom_fdim


@timeit(logger_name=TRAIN_LOGGER_NAME)
def cross_validate(args: TrainArgs,
                   train_func: Callable[[TrainArgs, MoleculeDataset, Logger], Dict[str, List[float]]]
                   ) -> Tuple[float, float]:
    """
    Runs k-fold cross-validation.

    For each of k splits (folds) of the data, trains and tests a model on that split
    and aggregates the performance across folds.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param train_func: Function which runs training.
    :return: A tuple containing the mean and standard deviation performance across folds.
    """
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    # Save args
    makedirs(args.save_dir)
    args.save(os.path.join(args.save_dir, 'args.json'))

    # Get data
    debug('Loading data')
    data = get_data(
        path=args.data_path,
        args=args,
        logger=logger,
        skip_none_targets=True,
    )
    validate_dataset_type(data, dataset_type=args.dataset_type)
    args.features_size = data.features_size()

    if args.atom_descriptors == 'descriptor':
        args.atom_descriptors_size = data.atom_descriptors_size()
        args.ffn_hidden_size += args.atom_descriptors_size
    elif args.atom_descriptors == 'feature':
        args.atom_features_size = data.atom_features_size()
        set_extra_atom_fdim(args.atom_features_size)

    debug(f'Number of tasks = {args.num_tasks}')

    # Run training on different random seeds for each fold
    all_scores = defaultdict(list)
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        data.reset_features_and_targets()
        model_scores = train_func(args, data, logger)
        for metric, scores in model_scores.items():
            all_scores[metric].append(scores)
    all_scores = dict(all_scores)

    # Convert scores to numpy arrays
    for metric, scores in all_scores.items():
        all_scores[metric] = np.array(scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num in range(args.num_folds):
        for metric, scores in all_scores.items():
            info(f'\tSeed {init_seed + fold_num} ==> test {metric} = {np.nanmean(scores[fold_num]):.6f}')

            if args.show_individual_scores:
                for task_name, score in zip(args.task_names, scores[fold_num]):
                    info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} {metric} = {score:.6f}')

    # Report scores across folds
    for metric, scores in all_scores.items():
        avg_scores = np.nanmean(scores, axis=1)  # average score for each model across tasks
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(args.task_names):
                info(f'\tOverall test {task_name} {metric} = '
                     f'{np.nanmean(scores[:, task_num]):.6f} +/- {np.nanstd(scores[:, task_num]):.6f}')

    # Save scores
    with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)

        header = ['Task']
        for metric in args.metrics:
            header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                      [f'Fold {i} {metric}' for i in range(args.num_folds)]
        writer.writerow(header)

        for task_num, task_name in enumerate(args.task_names):
            row = [task_name]
            for metric, scores in all_scores.items():
                task_scores = scores[:, task_num]
                mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
                row += [mean, std] + task_scores.tolist()
            writer.writerow(row)

    # Determine mean and std score of main metric
    avg_scores = np.nanmean(all_scores[args.metric], axis=1)
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)

    # Optionally merge and save test preds
    if args.save_preds:
        all_preds = pd.concat([pd.read_csv(os.path.join(save_dir, f'fold_{fold_num}', 'test_preds.csv'))
                               for fold_num in range(args.num_folds)])
        all_preds.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

    return mean_score, std_score


def chemprop_train() -> None:
    """Parses Chemprop training arguments and trains (cross-validates) a Chemprop model.

    This is the entry point for the command line command :code:`chemprop_train`.
    """
    cross_validate(args=TrainArgs().parse_args(), train_func=run_training)
