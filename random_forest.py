from argparse import ArgumentParser

from chemprop.parsing import add_train_args, modify_train_args
from chemprop.random_forest import cross_validate_random_forest
from chemprop.utils import create_logger


if __name__ == '__main__':
    parser = ArgumentParser()
    add_train_args(parser)
    parser.add_argument('--class_weight', type=str,
                        choices=['balanced'],
                        help='How to weight classes (None means no class balance)')
    parser.add_argument('--single_task', action='store_true', default=False,
                        help='Whether to run each task separately (needed when dataset has null entries)')
    parser.add_argument('--radius', type=int, default=2,
                        help='Morgan fingerprint radius')
    parser.add_argument('--num_bits', type=int, default=2048,
                        help='Number of bits in morgan fingerprint')
    parser.add_argument('--num_trees', type=int, default=500,
                        help='Number of random forest trees')
    args = parser.parse_args()
    modify_train_args(args)

    logger = create_logger(name='random_forest', save_dir=args.save_dir, quiet=args.quiet)

    if args.metric is None:
        if args.dataset_type == 'regression':
            args.metric = 'rmse'
        elif args.dataset_type == 'classification':
            args.metric = 'auc'
        else:
            raise ValueError(f'Default metric not supported for dataset_type "{args.dataset_type}"')

    cross_validate_random_forest(args, logger)
