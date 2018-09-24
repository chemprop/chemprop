import os
from pprint import pprint

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from mpn import build_MPN
from nn_utils import param_count
from parsing import parse_args
from train_utils import train, predict, evaluate, evaluate_predictions
from utils import get_data, get_loss_func, get_metric_func, split_data


def run_training(args) -> float:
    """Trains a model and returns test score on the model checkpoint with the highest validation score"""
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

    # Get loss and metric functions
    loss_func = get_loss_func(args.dataset_type)
    metric_func = get_metric_func(args.metric)

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Build/load model
        print('Building model {}'.format(model_idx))
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
        if args.checkpoint_paths is not None:
            print('Loading model from {}'.format(args.checkpoint_paths[model_idx]))
            model.load_state_dict(torch.load(args.checkpoint_paths[model_idx]))
        print(model)
        print('Number of parameters = {:,}'.format(param_count(model)))
        if args.cuda:
            print('Moving model to cuda')
            model = model.cuda()

        # Optimizer and learning rate scheduler
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        scheduler.step()

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        for _ in trange(args.epochs):
            print("Learning rate = {:.3e}".format(scheduler.get_lr()[0]))
            train(
                model=model,
                data=train_data,
                batch_size=args.batch_size,
                num_tasks=num_tasks,
                loss_func=loss_func,
                optimizer=optimizer,
                scaler=scaler,
                three_d=args.three_d
            )
            scheduler.step()
            val_score = evaluate(
                model=model,
                data=val_data,
                batch_size=args.batch_size,
                metric_func=metric_func,
                scaler=scaler,
                three_d=args.three_d
            )
            print('Validation {} = {:.3f}'.format(args.metric, val_score))

            # Save model checkpoint if improved validation score
            if args.minimize_score and val_score < best_score or \
                    not args.minimize_score and val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_{}.pt'.format(model_idx)))

    # Evaluate on test set
    smiles, labels = zip(*test_data)
    sum_preds = np.zeros((len(smiles), num_tasks))

    # Predict and evaluate each model individually
    for model_idx in range(args.ensemble_size):
        # Load state dict from best validation set performance
        model.load_state_dict(torch.load(os.path.join(args.save_dir + '/model_{}.pt'.format(model_idx))))

        model_preds = predict(
            model=model,
            smiles=smiles,
            batch_size=args.batch_size,
            scaler=scaler,
            three_d=args.three_d
        )
        model_score = evaluate_predictions(
            preds=model_preds,
            labels=labels,
            metric_func=metric_func
        )
        print('Model {} test {} = {:.3f}'.format(model_idx, args.metric, model_score))

        sum_preds += np.array(model_preds)

    # Evaluate ensemble
    avg_preds = sum_preds / args.ensemble_size
    ensemble_score = evaluate_predictions(
        preds=avg_preds.tolist(),
        labels=labels,
        metric_func=metric_func
    )
    print('Ensemble test {} = {:.3f}'.format(args.metric, ensemble_score))

    return ensemble_score


if __name__ == '__main__':
    run_training(parse_args())
