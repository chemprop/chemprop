from argparse import ArgumentParser, Namespace
import os
from tempfile import TemporaryDirectory

import torch


def get_parser():
    """Builds an argument parser"""
    parser = ArgumentParser()

    # General arguments
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--checkpoint_paths', type=str, nargs='*', default=None,
                        help='Paths to model checkpoints to load'
                             '(number of paths should match `ensemble_size` argument)')
    parser.add_argument('--dataset_type', type=str, choices=['classification', 'regression', 'regression_with_binning'],
                        help='Type of dataset, i.e. classification (cls) or regression (reg).'
                             'This determines the loss function used during training.')
    parser.add_argument('--num_bins', type=int, default=20,
                        help='Number of bins for regression with binning')
    parser.add_argument('--metric', type=str, default=None, choices=['auc', 'prc-auc', 'rmse', 'mae'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds when performing cross validation')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Gamma factor for exponential decay learning rate scheduler'
                             '(lr = gamma * lr)')
    parser.add_argument('--truncate_outliers', action='store_true', default=False,
                        help='Truncates outliers in the training set to improve training stability'
                             '(All values outside mean ± 3 * std are truncated to equal mean ± 3 * std)')

    # Model arguments
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh'],
                        help='Activation function')
    parser.add_argument('--attention', action='store_true', default=False,
                        help='Perform self attention over the atoms in a molecule')
    parser.add_argument('--message_attention', action='store_true', default=False,
                        help='Perform attention over messages.')
    parser.add_argument('--message_attention_heads', type=int, default=1,
                        help='Number of heads to use for message attention')
    parser.add_argument('--addHs', action='store_true', default=False,
                        help='Explicitly adds hydrogens to the molecular graph')
    parser.add_argument('--three_d', action='store_true', default=False,
                        help='Adds 3D coordinates to atom and bond features')
    parser.add_argument('--virtual_edges', action='store_true', default=False,
                        help='Adds virtual edges between non-bonded atoms')

    return parser


def modify_args(args: Namespace):
    """Modifies and validates arguments"""
    global temp_dir  # Prevents the temporary directory from being deleted upon function return

    # Argument modification/checking
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        temp_dir = TemporaryDirectory()
        args.save_dir = temp_dir.name

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    if args.metric is None:
        args.metric = 'auc' if args.dataset_type == 'classification' else 'rmse'

    if not (args.dataset_type == 'classification' and args.metric in ['auc', 'prc-auc'] or
            (args.dataset_type == 'regression' or args.dataset_type == 'regression_with_binning') and args.metric in ['rmse', 'mae']):
        raise ValueError('Metric "{}" invalid for dataset type "{}".'.format(args.metric, args.dataset_type))

    args.minimize_score = args.metric in ['rmse', 'mae']

    if args.checkpoint_paths is not None:
        assert len(args.checkpoint_paths) == args.ensemble_size


def parse_args():
    """Parses arguments (includes modifying/validating arguments)"""
    parser = get_parser()
    args = parser.parse_args()
    modify_args(args)

    return args
