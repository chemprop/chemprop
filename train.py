import logging

from chemprop.parsing import parse_train_args
from chemprop.train import cross_validate
from chemprop.utils import set_logger


# Initialize logger
logger = logging.getLogger('train')


if __name__ == '__main__':
    args = parse_train_args()
    set_logger(logger, args.save_dir, args.quiet)
    cross_validate(args, logger)
