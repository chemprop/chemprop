from argparse import ArgumentParser
import math
import os
from pprint import pprint

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from mpn import build_MPN
from nnutils import param_count
from train_utils import train, evaluate
from utils import get_data, get_loss_func, get_metric_func, split_data


def main(args):
    pprint(vars(args))

    print('Loading data')
    data = get_data(args.data_path)
    train_data, val_data, test_data = split_data(data, seed=args.seed)
    num_tasks = len(data[0][1])

    print('Train size = {:,}, val size = {:,}, test size = {:,}'.format(
        len(train_data),
        len(val_data),
        len(test_data))
    )
    print('Number of tasks = {}'.format(num_tasks))
    
    # Initialize scaler which subtracts mean and divides by standard deviation
    if args.scale:
        print('Fitting scaler')
        train_labels = list(zip(*train_data))[1]
        scaler = StandardScaler().fit(train_labels)
    else:
        scaler = None

    print('Building model')
    model = build_MPN(
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_tasks=args.num_tasks,
        sigmoid=args.dataset == 'classification',
        dropout=args.dropout,
        activation=args.activation
    )
    print(model)
    print('Number of parameters = {:,}'.format(param_count(model)))
    if args.cuda:
        print('Moving model to cuda')
        model = model.cuda()

    # Get loss and metric functions
    loss_func = get_loss_func(args.dataset_type)
    metric_func = get_metric_func(args.metric)

    # Optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    scheduler.step()

    # Run training
    best_loss = float('inf')
    for epoch in trange(args.epochs):
        print("Learning rate = {:.3e}".format(scheduler.get_lr()[0]))

        train(
            train_data,
            args.batch_size,
            num_tasks,
            model,
            loss_func,
            optimizer,
            scaler
        )
        scheduler.step()
        val_loss, val_metric = evaluate(
            val_data,
            args.batch_size,
            num_tasks,
            model,
            metric_func,
            scaler
        )
        print('Validation loss = {:.3f}'.format(val_loss))

        # Save model checkpoints
        if args.save_dir is not None:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model.iter-" + str(epoch)))

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, "model.best"))


if __name__ == '__main__':
    parser = ArgumentParser()

    # General arguments
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
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
    parser.add_argument('--scale', action='store_true', default=False,
                        help='Scale labels by subtracting mean and dividing by standard deviation'
                             '(useful for qm regression datasets)')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh'],
                        help='Activation function')

    args = parser.parse_args()

    # Argument modification/checking
    os.makedirs(args.save_dir, exist_ok=True)
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.metric is None:
        args.metric = 'roc' if args.dataset_type == 'classification' else 'rmse'

    if not (args.dataset_type == 'classification' and args.metric in ['roc', 'prc-auc'] or
            args.dataset_type == 'regression' and args.metric in ['rmse', 'mae']):
        raise ValueError('Metric "{}" invalid for dataset type "{}".'.format(args.metric, args.dataset_type))

    main(args)
