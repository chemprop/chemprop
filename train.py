import os
from pprint import pprint

from sklearn.preprocessing import StandardScaler
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from mpn import build_MPN
from nn_utils import param_count
from parsing import parse_args
from train_utils import train, evaluate
from utils import get_data, get_loss_func, get_metric_func, split_data


def run_training(args) -> float:
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

    # Build/load model
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
    if args.checkpoint_path is not None:
        print('Loading model from {}'.format(args.checkpoint_path))
        model.load_state_dict(torch.load(args.checkpoint_path))
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

    return test_score


if __name__ == '__main__':
    run_training(parse_args())
