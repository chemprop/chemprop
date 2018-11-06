from argparse import ArgumentParser, Namespace
import logging
from pprint import pprint

from chemprop.data_processing import average_duplicates, resplit
from chemprop.parsing import add_hyper_opt_args, add_predict_args, add_train_args, modify_hyper_opt_args,\
    modify_train_args, update_args_from_checkpoint_dir
from chemprop.train import cross_validate
from chemprop.utils import set_logger
from hyper_opt import load_sorted_results, optimize_hyperparameters
from predict import make_predictions


# Initialize logger
logger = logging.getLogger('end_to_end')
logger.setLevel(logging.DEBUG)


def merge_train_val(args: Namespace):
    with open(args.train_save, 'r') as tf, \
            open(args.val_save, 'r') as vf, \
            open(args.train_val_save, 'w') as tvf:
        for line in tf:
            tvf.write(line.strip() + '\n')
        vf.readline()  # skip header from validation file
        for line in vf:
            tvf.write(line.strip() + '\n')


if __name__ == '__main__':
    # Note: Just put junk for --data_path, it's required but it'll be overwritten
    # Also need to specify:
    # --dataset_type "classification" or "regression"
    # --results_dir for hyperparam results
    # --test_path for test csv file
    # --preds_path where predictions will be saved
    # --compound_names if necessary - might need to hard code so only during test
    parser = ArgumentParser()
    add_train_args(parser)
    add_hyper_opt_args(parser)
    add_predict_args(parser)
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to CSV file containing training data in chronological order')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to CSV file containing val data in chronological order')
    parser.add_argument('--train_save', type=str, required=True,
                        help='Path to CSV file for new train data')
    parser.add_argument('--val_save', type=str, required=True,
                        help='Path to CSV file for new val data')
    parser.add_argument('--val_frac', type=float, default=0.2,
                        help='frac of data to use for validation')
    parser.add_argument('--train_val_save', type=str, required=True,
                        help='Path to CSV file for combined train and val data')
    args = parser.parse_args()

    set_logger(logger, args.save_dir, args.quiet)
    modify_train_args(args)
    modify_hyper_opt_args(args)

    # Preprocess train and validation data
    resplit(args)
    merge_train_val(args)
    for path in [args.train_save, args.val_save, args.train_val_save]:
        args.data_path = path
        args.save_path = path
        average_duplicates(args)

    # Optimize hyperparameters
    args.data_path = args.train_save
    args.separate_test_set = args.val_save
    optimize_hyperparameters(args)

    # Determine best hyperparameters, update args, and train
    results = load_sorted_results(args.results_dir)
    config = results[0]
    config.pop('loss')
    print('Best config')
    pprint(config)
    for key, value in config.items():
        setattr(args, key, value)

    args.data_path = args.train_val_save
    args.separate_test_set = None
    args.split_sizes = [0.8, 0.2, 0.0]  # no need for a test set during training

    cross_validate(args, logger)

    # Predict on test data
    args.checkpoint_dir = args.save_dir
    update_args_from_checkpoint_dir(args)
    args.compound_names = True  # only if test set has compound names
    args.ensemble_size = 5  # might want to make this an arg somehow (w/o affecting hyperparameter optimization)

    make_predictions(args)
