from argparse import ArgumentParser, Namespace
import json
import os
from pprint import pprint

from parsing import add_hyper_opt_args, add_predict_args, add_train_args, modify_hyper_opt_args, modify_train_args,\
    update_args_from_checkpoint_dir
from train import cross_validate
from hyper_opt import optimize_hyperparameters
from resplit_data import resplit
from avg_dups import average_duplicates
from predict import make_predictions


def merge_train_val(args: Namespace):
    with open(args.train_save, 'r') as tf, \
            open(args.val_save, 'r') as vf, \
            open(args.train_val_save, 'w') as tvf:
        for line in tf:
            tvf.write(line.strip() + '\n')
        vf.readline()  # skip header from validation file
        for line in vf:
            tvf.write(line.strip() + '\n')


def update_args_from_best_config(args: Namespace):
    # Find best config index from results.json
    best_config_index = None
    best_loss = float('inf')
    with open(os.path.join(args.results_dir, 'results.json'), 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            config_index, loss = result[0], result[3]['loss']
            if loss < best_loss:
                best_config_index, best_loss = config_index, loss

    # Use best config index to identify best config from configs.json
    with open(os.path.join(args.results_dir, 'configs.json'), 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            config_index, config = result[0], result[1]
            if config_index == best_config_index:
                print('Best config')
                pprint(config)
                for key, value in config.items():
                    setattr(args, key, value)
                break


if __name__ == '__main__':
    # Note: Just put junk for --data_path, it's required but it'll be overwritten
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
    parser.add_argument('--hyperopt_timeout', type=float, default=-1,
                        help='seconds to wait for hyperopt script')
    args = parser.parse_args()

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

    # Determine best hyperparameters and train
    args.data_path = args.train_val_save
    args.separate_test_set = None
    args.split_sizes = [0.8, 0.2, 0.0]  # no need for a test set during training
    update_args_from_best_config(args)
    cross_validate(args)

    # Predict on test data
    args.checkpoint_dir = args.save_dir
    update_args_from_checkpoint_dir(args)
    make_predictions(args)
