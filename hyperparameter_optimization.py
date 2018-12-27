from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import Dict, Union

from hyperopt import fmin, hp, tpe

from chemprop.models import build_model
from chemprop.nn_utils import param_count
from chemprop.parsing import add_train_args, modify_train_args
from chemprop.train import cross_validate
from model_comparison import create_logger, create_train_logger, DATASETS


SPACE = {
    'hidden_size': hp.quniform('hidden_size', low=300, high=2400, q=100),
    'depth': hp.quniform('depth', low=2, high=6, q=1),
    'dropout': hp.quniform('dropout', low=0.0, high=0.4, q=0.05),
    'ffn_num_layers': hp.quniform('ffn_num_layers', low=1, high=3, q=1)
}
INT_KEYS = ['hidden_size', 'depth', 'ffn_num_layers']

TRAIN_LOGGER = create_train_logger()


def grid_search(args: Namespace):
    for dataset_name in args.datasets:
        # Get dataset
        dataset_type, dataset_path, _, metric = DATASETS[dataset_name]

        # Create logger for dataset
        logger = create_logger(name=dataset_name,
                               save_dir=args.save_dir,
                               save_name='{}_{}.log'.format(dataset_name, args.split_type))

        # Set up args for dataset
        dataset_args = deepcopy(args)
        dataset_args.data_path = dataset_path
        dataset_args.dataset_type = dataset_type
        dataset_args.save_dir = None
        dataset_args.metric = metric
        modify_train_args(dataset_args)

        # Run grid search
        results = []

        # Define hyperparameter optimization
        def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
            # Convert hyperparms from float to int when necessary
            for key in INT_KEYS:
                hyperparams[key] = int(hyperparams[key])

            # Copy args
            gs_args = deepcopy(dataset_args)

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

            return (1 if gs_args.minimize_score else -1) * mean_score

        fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_runs_per_dataset)

        # Report best result
        best_result = min(results, key=lambda result: (1 if dataset_args.minimize_score else -1) * result['mean_score'])
        logger.info('best')
        logger.info(best_result['hyperparams'])
        logger.info('num params: {:,}'.format(best_result['num_params']))
        logger.info('{} +/- {} {}'.format(best_result['mean_score'], best_result['std_score'], metric))


if __name__ == '__main__':
    parser = ArgumentParser()
    add_train_args(parser)
    parser.add_argument('--datasets', type=str, nargs='+', default=list(DATASETS.keys()), choices=list(DATASETS.keys()),
                        help='Which datasets to perform a grid search on')
    parser.add_argument('--num_runs_per_dataset', type=int, default=20,
                        help='Number of hyperparameter choices to try for each datasets')
    args = parser.parse_args()

    grid_search(args)

    # python hyperparameter_optimization.py --data_path blah --dataset_type regression --save_dir ../chemprop_grid_search --datasets delaney --num_runs_per_dataset 50 --num_folds 3 --quiet
