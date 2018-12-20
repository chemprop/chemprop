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
        hyperparams[param] = random.sample(options)

    return hyperparams


def grid_search(args: Namespace):
    for dataset_name in args.datasets:
        # Get dataset
        dataset_type, dataset_path, num_folds, metric = DATASETS[dataset_name]

        # Create logger for dataset
        logger = create_logger(name=dataset_name, save_path=os.path.join(args.save_dir, dataset_name + '.log'))

        # Set up args for dataset
        dataset_args = deepcopy(args)
        dataset_args.data_path = dataset_path
        dataset_args.dataset_type = dataset_type
        dataset_args.save_dir = os.path.join(args.save_dir, dataset_name)
        dataset_args.num_folds = num_folds
        dataset_args.metric = metric
        modify_train_args(dataset_args)

        # Run grid search
        results = {}
        for _ in range(args.num_runs_per_dataset):
            # Set up args for hyperparameter choices
            gs_args = deepcopy(dataset_args)

            hyperparams = sample_hyperparams(GRID)
            for key, value in hyperparams:
                setattr(gs_args, key, value)

            # Set up logging for training
            os.makedirs(args.save_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(args.save_dir, args.log_name))
            fh.setLevel(logging.DEBUG)

            # Cross validate
            TRAIN_LOGGER.addHandler(fh)
            mean_score, std_score = cross_validate(args, TRAIN_LOGGER)
            TRAIN_LOGGER.removeHandler(fh)

            # Record results
            logger.info(hyperparams)
            temp_model = build_model(args)
            logger.info('num params: {:,}'.format(param_count(temp_model)))
            logger.info('{} +/- {} {}'.format(mean_score, std_score, metric))

            results[mean_score]

if __name__ == '__main__':
    parser = ArgumentParser()
    add_train_args(parser)
    parser.add_argument('--results_name', type=str, default='grid_search.csv',
                        help='Name of file where grid search results will be saved')
    parser.add_argument('--datasets', type=str, nargs='+', default=['all'], choices=DATASETS.keys(),
                        help='Which datasets to perform a grid search on')
    parser.add_argument('--num_runs_per_dataset', type=int, default=10,
                        help='Number of hyperparameter choices to try for each datasets')
    args = parser.parse_args()

    grid_search(args)

    # python model_comparison.py --data_path blah --dataset_type regression --save_dir logging_dir --log_name grid_search.csv --quiet
