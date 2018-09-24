from argparse import ArgumentParser
import os
from tempfile import TemporaryDirectory

import torch


def get_parser():
    parser = ArgumentParser()

    # General arguments
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint to load')
    parser.add_argument('--dataset_type', type=str, choices=['classification', 'regression'],
                        help='Type of dataset, i.e. classification (cls) or regression (reg).'
                             'This determines the loss function used during training.')
    parser.add_argument('--metric', type=str, default=None, choices=['roc', 'prc-auc', 'rmse', 'mae'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "roc" for classification and "rmse" for regression.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets')
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

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh'],
                        help='Activation function')
    parser.add_argument('--attention', action='store_true', default=False,
                        help='Perform self attention over the atoms in a molecule.')
    parser.add_argument('--three_d', action='store_true', default=False,
                        help='Adds 3D coordinates to atom and bond features')

    return parser


def modify_args(args):
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
        args.metric = 'roc' if args.dataset_type == 'classification' else 'rmse'

    if not (args.dataset_type == 'classification' and args.metric in ['roc', 'prc-auc'] or
            args.dataset_type == 'regression' and args.metric in ['rmse', 'mae']):
        raise ValueError('Metric "{}" invalid for dataset type "{}".'.format(args.metric, args.dataset_type))

    args.minimize_score = args.metric in ['rmse', 'mae']


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    modify_args(args)

    return args
