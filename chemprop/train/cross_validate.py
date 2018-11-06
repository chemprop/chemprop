from argparse import Namespace
from logging import Logger
import os

import numpy as np
import shutil

from chemprop.train import run_training
from chemprop.utils.utils import get_task_names, get_desired_labels


def cross_validate(args: Namespace, logger: Logger):
    """k-fold cross validation"""
    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)
    desired_labels = get_desired_labels(args, task_names)

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        logger.info('Fold {}'.format(fold_num))
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, 'fold_{}'.format(fold_num))
        os.makedirs(args.save_dir, exist_ok=True)
        model_scores = run_training(args)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report results
    logger.info('{}-fold cross validation'.format(args.num_folds))

    # Report scores for each model
    for fold_num, scores in enumerate(all_scores):
        logger.info('Seed {} ==> test {} = {:.3f}'.format(init_seed + fold_num, args.metric, np.mean(scores)))

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                if task_name in desired_labels:
                    logger.info('Seed {} ==> test {} {} = {:.3f}'.format(init_seed + fold_num, task_name, args.metric, score))

    # Report scores across models
    avg_scores = np.mean(all_scores, axis=1)  # average score for each model across tasks
    logger.info('Overall test {} = {:.3f} ± {:.3f}'.format(args.metric, np.mean(avg_scores), np.std(avg_scores)))

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            if task_name in desired_labels:
                logger.info('Overall test {} {} = {:.3f} +/- {:.3f}'.format(
                    task_name,
                    args.metric,
                    np.mean(all_scores[:, task_num]),
                    np.std(all_scores[:, task_num]))
                )

    if args.num_chunks > 1:
        shutil.rmtree(args.chunk_temp_dir)
