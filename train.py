from argparse import ArgumentParser
import os
from pprint import pprint
from tempfile import TemporaryDirectory

from sklearn.preprocessing import StandardScaler
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from mpn import build_MPN
from nn_utils import param_count
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
    
    # Initialize scaler which subtracts mean and divides by standard deviation for regression datasets
    if args.dataset_type == 'regression':
        print('Fitting scaler')
        train_labels = list(zip(*train_data))[1]
        scaler = StandardScaler().fit(train_labels)
    else:
        scaler = None

    print('Building model')
    model = build_MPN(
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_tasks=num_tasks,
        sigmoid=args.dataset_type == 'classification',
        dropout=args.dropout,
        activation=args.activation,
        attention=args.attention,
        three_d=args.three_d
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
    best_score = float('inf') if args.minimize_score else -float('inf')
    for epoch in trange(args.epochs):
        print("Learning rate = {:.3e}".format(scheduler.get_lr()[0]))

        train(
            data=train_data,
            batch_size=args.batch_size,
            num_tasks=num_tasks,
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scaler=scaler,
            three_d=args.three_d
        )
        scheduler.step()
        val_score = evaluate(
            data=val_data,
            batch_size=args.batch_size,
            num_tasks=num_tasks,
            model=model,
            metric_func=metric_func,
            scaler=scaler,
            three_d=args.three_d
        )
        print('Validation {} = {:.3f}'.format(args.metric, val_score))

        # Save model checkpoints
        if args.save_dir is not None:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model.iter-" + str(epoch)))

            if args.minimize_score and val_score < best_score or \
                    not args.minimize_score and val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), os.path.join(args.save_dir, "model.best"))

    # Evaluate on test set
    model.load_state_dict(torch.load(os.path.join(args.save_dir + "/model.best")))
    test_score = evaluate(
        data=test_data,
        batch_size=args.batch_size,
        num_tasks=num_tasks,
        model=model,
        metric_func=metric_func,
        scaler=scaler,
        three_d=args.three_d
    )
    print("Test {} = {:.3f}".format(args.metric, test_score))


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

    args = parser.parse_args()

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

    main(args)
