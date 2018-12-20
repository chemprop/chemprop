from argparse import ArgumentParser, Namespace
from copy import deepcopy
import logging
import os
import random
from typing import Dict, List, Union

from chemprop.models import build_model
from chemprop.nn_utils import param_count
from chemprop.parsing import add_train_args, modify_train_args
from chemprop.train import cross_validate
from model_comparison import create_logger, create_train_logger, DATASETS


GRID = {
    'hidden_size': [300, 600, 1200, 2400],
    'depth': [3, 4, 5, 6],
    'dropout': [0.0, 0.2, 0.4],
    'ffn_num_layers': [1, 2, 3],
    # 'ffn_hidden_size_multiplier': [0.5, 1.0]
}

TRAIN_LOGGER = create_train_logger()


def sample_hyperparams(grid: Dict[str, Union[List[int], List[float]]]) -> Dict[str, Union[int, float]]:
    hyperparams = {}
    for param, options in grid.items():
        hyperparams[param] = random.choice(options)

    return hyperparams


def grid_search(args: Namespace):
    for dataset_name in args.datasets:
        # Get dataset
        dataset_type, dataset_path, num_folds, metric = DATASETS[dataset_name]

        # Create logger for dataset
        logger = create_logger(name=dataset_name, save_dir=args.save_dir, save_name='{}.log'.format(dataset_name))

        # Set up args for dataset
        dataset_args = deepcopy(args)
        dataset_args.data_path = dataset_path
        dataset_args.dataset_type = dataset_type
        dataset_args.save_dir = os.path.join(args.save_dir, dataset_name)
        dataset_args.num_folds = num_folds
        dataset_args.metric = metric
        modify_train_args(dataset_args)

        # Run grid search
        results = []
        for _ in range(args.num_runs_per_dataset):
            # Set up args for hyperparameter choices
            gs_args = deepcopy(dataset_args)

            hyperparams = sample_hyperparams(GRID)
            for key, value in hyperparams.items():
                setattr(gs_args, key, value)

            # Cross validate
            mean_score, std_score = cross_validate(gs_args, TRAIN_LOGGER)

            # Record results
            temp_model = build_model(gs_args)
            num_params = param_count(temp_model)
            logger.info(hyperparams)
            logger.info('num params: {:,}'.format(num_params))
            logger.info('{} +/- {} {}'.format(mean_score, std_score, metric))

            results.append({
                'mean_score': mean_score,
                'std_score': std_score,
                'hyperparams': hyperparams,
                'num_params': num_params
            })

        # Report best result
        results.sort(key=lambda result: (1 if dataset_args.minimize_score else -1) * result['mean_score'])
        best_result = results[0]
        logger.info('best')
        logger.info(best_result['hyperparams'])
        logger.info('num params: {:,}'.format(best_result['num_params']))
        logger.info('{} +/- {} {}'.format(best_result['mean_score'], best_result['std_score'], metric))


if __name__ == '__main__':
    parser = ArgumentParser()
    add_train_args(parser)
    parser.add_argument('--datasets', type=str, nargs='+', default=['all'], choices=DATASETS.keys(),
                        help='Which datasets to perform a grid search on')
    parser.add_argument('--num_runs_per_dataset', type=int, default=10,
                        help='Number of hyperparameter choices to try for each datasets')
    args = parser.parse_args()

    grid_search(args)

    # python model_comparison.py --data_path blah --dataset_type regression --save_dir logging_dir --datasets all --num_runs_per_dataset 10 --quiet
