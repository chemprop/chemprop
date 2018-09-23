from argparse import ArgumentParser
import math
import os
from pprint import pprint
from typing import Callable, List, Tuple

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from mpn import mol2graph, MPN
from nnutils import param_count
from utils import get_data, split_data


def train(data: List[Tuple[str, List[float]]],
          batch_size: int,
          num_tasks: int,
          model: nn.Module,
          loss_fn: Callable,
          optimizer: Adam,
          scaler: StandardScaler = None):
    """
    Trains a model for an epoch.

    :param data: Training data.
    :param batch_size: Batch size.
    :param num_tasks: Number of tasks.
    :param model: Model.
    :param loss_fn: Loss function.
    :param optimizer: Optimizer.
    :param scaler: A StandardScaler object fit on the training labels.
    """
    model.train()

    loss_sum, num_iter = 0, 0
    for i in range(0, len(data), batch_size):
        # Prepare batch
        batch = data[i:i + batch_size]
        mol_batch, label_batch = zip(*batch)
        mol_batch = mol2graph(mol_batch)
        mask = torch.Tensor([[x is not None for x in lb] for lb in label_batch])
        labels = [[0 if x is None else x for x in lb] for lb in label_batch]
        if scaler is not None:
            labels = scaler.transform(labels)  # subtract mean, divide by std
        labels = torch.Tensor(labels)

        # Run model
        model.zero_grad()
        preds = model(mol_batch)
        loss = loss_fn(preds, labels) * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item() * batch_size
        num_iter += batch_size

        loss = loss * num_tasks
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            pnorm = math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))
            gnorm = math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters()]))
            print("Loss = {:.4f}, PNorm = {:.4f}, GNorm = {:.4f}".format(math.sqrt(loss_sum / num_iter), pnorm, gnorm))
            loss_sum, num_iter = 0, 0


def evaluate(data: List[Tuple[str, List[float]]],
             batch_size: int,
             num_tasks: int,
             model: nn.Module,
             metric: Callable,
             scaler: StandardScaler = None) -> float:
    """
    Evaluates a model on a dataset.

    :param data: Validation dataset.
    :param batch_size: Batch size.
    :param num_tasks: Number of tasks.
    :param model: Model.
    :param metric: Metric function which takes in a list of labels and a list of predictions.
    :param scaler: A StandardScaler object fit on the training labels.
    :return: Root mean square error.
    """
    model.eval()

    all_preds = [[] for _ in range(num_tasks)]
    all_labels = [[] for _ in range(num_tasks)]
    for i in range(0, len(data), batch_size):
        # Prepare batch
        batch = data[i:i + batch_size]
        mol_batch, label_batch = zip(*batch)
        mol_batch = mol2graph(mol_batch)
        mask = torch.Tensor([[x is not None for x in lb] for lb in label_batch])
        labels = [[0 if x is None else x for x in lb] for lb in label_batch]

        # Run model
        preds = model(mol_batch)
        preds = preds.data.cpu().numpy()
        if scaler is not None:
            preds = scaler.inverse_transform(preds)

        # Collect predictions and labels, skipping those without labels (i.e. masked out)
        for i in range(num_tasks):
            for j in range(len(batch)):
                if mask[j][i] == 1:
                    all_preds[i].append(preds[j][i])
                    all_labels[i].append(labels[j][i])

    # Compute metric
    results = []
    for i in range(num_tasks):
        # Skip if all labels are identical
        if all(label == 0 for label in all_labels[i]) or all(label == 1 for label in all_labels[i]):
            continue
        results.append(metric(all_labels[i], all_preds[i]))

    # Average
    result = sum(results) / len(results)

    return result


def main(args):
    pprint(vars(args))

    print('Loading data')
    data = get_data(args.data_path)
    train_data, val_data, test_data = split_data(data, seed=args.seed)
    num_tasks = len(data[0][1])

    # Initialize scaler which subtracts mean and divides by standard deviation
    if args.scale:
        train_labels = list(zip(*train_data))[1]
        scaler = StandardScaler().fit(train_labels)
    else:
        scaler = None

    print('Train size = {:,}, val size = {:,}, test size = {:,}'.format(
        len(train_data),
        len(val_data),
        len(test_data))
    )

    print('Number of tasks = {}'.format(num_tasks))

    print('Building model')
    encoder = MPN(
        hidden_size=args.hidden_size,
        depth=args.depth,
        dropout=args.dropout,
        act_func=args.act_func
    )
    modules = [
        encoder,
        nn.Linear(args.hidden_size, args.hidden_size),
        nn.ReLU(),
        nn.Linear(args.hidden_size, num_tasks)
    ]
    if args.dataset_type == 'classification':
        modules.append(nn.Sigmoid())
    model = nn.Sequential(*modules)

    if args.cuda:
        model = model.cuda()

    # Initialize parameters
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    print(model)
    print('Number of parameters = {:,}'.format(param_count(model)))

    # Loss function
    if args.dataset_type == 'classification':
        loss_fn = nn.BCELoss(reduction='none')
    elif args.dataset_type == 'regression':
        loss_fn = nn.MSELoss(reduction='none')
    else:
        raise ValueError('Dataset type "{}" not supported.'.format(args.dataset_type))

    # Metric function
    if args.metric == 'roc':
        metric = roc_auc_score
    elif args.metric == 'prc-auc':
        def metric(labels, preds):
            precision, recall, _ = precision_recall_curve(labels, preds)
            return auc(recall, precision)
    elif args.metric == 'rmse':
        metric = lambda labels, preds: math.sqrt(mean_squared_error(labels, preds))
    elif args.metric == 'mae':
        metric = mean_absolute_error
    else:
        raise ValueError('Metric "{}" not supported.'.format(args.metric))

    # Optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    scheduler.step()

    best_loss = float('inf')
    for epoch in trange(args.epochs):
        print("Learning rate = {:.3e}".format(scheduler.get_lr()[0]))

        train(
            train_data,
            args.batch_size,
            num_tasks,
            model,
            loss_fn,
            optimizer,
            scaler
        )
        scheduler.step()
        val_loss, val_metric = evaluate(
            val_data,
            args.batch_size,
            num_tasks,
            model,
            metric,
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
    parser.add_argument('--dataset_type', type=str, choices=['classification', 'cls', 'regression', 'reg'],
                        help='Type of dataset, i.e. classification (cls) or regression (reg).'
                             'This determines the loss function used during training.')
    parser.add_argument('--metric', type=str, choices=['roc', 'prc-auc', 'rmse', 'mae'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).')
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
                        help='Dropout rate')
    parser.add_argument('--act_func', type=str, default='ReLU', choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh'],
                        help='Activation function')

    args = parser.parse_args()

    # Argument modification/checking
    os.makedirs(args.save_dir, exist_ok=True)
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.dataset_type == 'cls':
        args.dataset_type = 'classification'
    elif args.dataset_type == 'reg':
        args.dataset_type = 'regression'

    if not (args.dataset_type == 'classification' and args.metric in ['roc', 'prc-auc'] or
            args.dataset_type == 'regression' and args.metric in ['rmse', 'mae']):
        raise ValueError('Metric "{}" invalid for dataset type "{}".'.format(args.metric, args.dataset_type))

    main(args)
