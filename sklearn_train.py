from chemprop.args import SklearnTrainArgs
from chemprop.sklearn_train import cross_validate_sklearn
from chemprop.utils import create_logger


if __name__ == '__main__':
    args = SklearnTrainArgs().parse_args()

    logger = create_logger(name='sklearn-train', save_dir=args.save_dir, quiet=args.quiet)

    if args.metric is None:
        if args.dataset_type == 'regression':
            args.metric = 'rmse'
        elif args.dataset_type == 'classification':
            args.metric = 'auc'
        else:
            raise ValueError(f'Default metric not supported for dataset_type "{args.dataset_type}"')

    cross_validate_sklearn(args, logger)
