import csv
from logging import Logger
import os
from typing import Tuple

import numpy as np

from .run_training import run_training
from chemprop.args import TrainArgs
from chemprop.data.utils import get_task_names
from chemprop.utils import create_logger, makedirs


def cross_validate(args: TrainArgs, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    args.task_names = get_task_names(
        path=args.data_path,
        smiles_column=args.smiles_column,
        target_columns=args.target_columns,
        ignore_columns=args.ignore_columns
    )

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores = run_training(args, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(args.task_names, scores):
                info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(args.task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    # Save scores
    with open(os.path.join(save_dir, 'test_scores.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', f'Mean {args.metric}', f'Standard deviation {args.metric}'] +
                        [f'Fold {i} {args.metric}' for i in range(args.num_folds)])

        for task_num, task_name in enumerate(args.task_names):
            task_scores = all_scores[:, task_num]
            mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
            writer.writerow([task_name, mean, std] + task_scores.tolist())

    return mean_score, std_score


def chemprop_train() -> None:
    """Runs chemprop training."""
    args = TrainArgs().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    cross_validate(args, logger)
