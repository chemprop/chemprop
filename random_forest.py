from argparse import ArgumentParser

from chemprop.random_forest import cross_validate_random_forest
from chemprop.utils import create_logger


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV')
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['regression', 'classification'],
                        help='Dataset type')
    parser.add_argument('--save_dir', type=str,
                        help='Directory where results will be saved')
    parser.add_argument('--metric', type=str,
                        choices=['auc', 'prc-auc', 'rmse', 'mae'],
                        help='Metric to use during evaluation.')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'crossval'],
                        help='Split type')
    parser.add_argument('--class_weight', type=str,
                        choices=['balanced'],
                        help='How to weight classes (None means no class balance)')
    parser.add_argument('--single_task', action='store_true', default=False,
                        help='Whether to run each task separately (needed when dataset has null entries)')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds of cross validation')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--radius', type=int, default=2,
                        help='Morgan fingerprint radius')
    parser.add_argument('--num_bits', type=int, default=2048,
                        help='Number of bits in morgan fingerprint')
    parser.add_argument('--num_trees', type=int, default=500,
                        help='Number of random forest trees')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Control verbosity level')
    args = parser.parse_args()

    logger = create_logger(name='random_forest', save_dir=args.save_dir, quiet=args.quiet)

    if args.metric is None:
        if args.dataset_type == 'regression':
            args.metric = 'rmse'
        elif args.dataset_type == 'classification':
            args.metric = 'auc'
        else:
            raise ValueError(f'Default metric not supported for dataset_type "{args.dataset_type}"')

    cross_validate_random_forest(args, logger)
