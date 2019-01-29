"""Trains a model on a dataset."""

from chemprop.parsing import parse_train_args
from chemprop.train import cross_validate
from chemprop.utils import create_logger


if __name__ == '__main__':
    args = parse_train_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    cross_validate(args, logger)
