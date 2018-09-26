from argparse import Namespace
import logging
import os
from pprint import pformat

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from mpn import build_MPN
from nn_utils import param_count
from parsing import parse_args
from train_utils import train, predict, evaluate, evaluate_predictions
from utils import get_data, get_loss_func, get_metric_func, set_logger, split_data


# Initialize logger
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)


def run_training(args: Namespace) -> float:
    """Trains a model and returns test score on the model checkpoint with the highest validation score"""
    logger.debug(pformat(vars(args)))

    logger.debug('Loading data')
    data = get_data(args.data_path)
    logger.debug('Splitting data with seed {}'.format(args.seed))
    train_data, val_data, test_data = split_data(data, seed=args.seed)
    num_tasks = len(data[0][1])

    logger.debug('Train size = {:,} | val size = {:,} | test size = {:,}'.format(
        len(train_data),
        len(val_data),
        len(test_data))
    )
    logger.debug('Number of tasks = {}'.format(num_tasks))
    
    # Initialize scaler which subtracts mean and divides by standard deviation for regression datasets
    if args.dataset_type == 'regression':
        logger.debug('Fitting scaler')
        train_labels = list(zip(*train_data))[1]
        scaler = StandardScaler().fit(train_labels)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args.dataset_type)
    metric_func = get_metric_func(args.metric)

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, 'model_{}'.format(model_idx))
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=save_dir)

        # Build/load model
        logger.debug('Building model {}'.format(model_idx))
        model = build_MPN(num_tasks, args)

        if args.checkpoint_paths is not None:
            logger.debug('Loading model from {}'.format(args.checkpoint_paths[model_idx]))
            model.load_state_dict(torch.load(args.checkpoint_paths[model_idx]))
            # TODO: maybe remove the line below - it's a hack to ensure that you can evaluate
            # on test set if training for 0 epochs
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

        logger.debug(model)
        logger.debug('Number of parameters = {:,}'.format(param_count(model)))
        if args.cuda:
            logger.debug('Moving model to cuda')
            model = model.cuda()

        # Optimizer and learning rate scheduler
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        scheduler.step()

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            logger.debug('Epoch {}'.format(epoch))
            lr = scheduler.get_lr()[0]
            logger.debug("Learning rate = {:.3e}".format(lr))
            writer.add_scalar('learning_rate', lr, n_iter)

            n_iter = train(
                model=model,
                data=train_data,
                n_iter=n_iter,
                loss_func=loss_func,
                optimizer=optimizer,
                args=args,
                scaler=scaler,
                logger=logger,
                writer=writer
            )
            scheduler.step()
            val_score = evaluate(
                model=model,
                data=val_data,
                metric_func=metric_func,
                args=args,
                scaler=scaler
            )

            logger.debug('Validation {} = {:.3f}'.format(args.metric, val_score))
            writer.add_scalar('validation_{}'.format(args.metric), val_score, n_iter)

            # Save model checkpoint if improved validation score
            if args.minimize_score and val_score < best_score or \
                    not args.minimize_score and val_score > best_score:
                best_score, best_epoch = val_score, epoch
                torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

        logger.info('Model {} best validation {} = {:.3f} on epoch {}'.format(model_idx, args.metric, best_score, best_epoch))

    # Evaluate on test set
    smiles, labels = zip(*test_data)
    sum_preds = np.zeros((len(smiles), num_tasks))

    # Predict and evaluate each model individually
    for model_idx in range(args.ensemble_size):
        # Load state dict from best validation set performance
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_{}/model.pt'.format(model_idx))))

        model_preds = predict(
            model=model,
            smiles=smiles,
            args=args,
            scaler=scaler
        )
        model_score = evaluate_predictions(
            preds=model_preds,
            labels=labels,
            metric_func=metric_func
        )
        logger.info('Model {} test {} = {:.3f}'.format(model_idx, args.metric, model_score))

        sum_preds += np.array(model_preds)

    # Evaluate ensemble
    avg_preds = sum_preds / args.ensemble_size
    ensemble_score = evaluate_predictions(
        preds=avg_preds.tolist(),
        labels=labels,
        metric_func=metric_func
    )
    logger.info('Ensemble test {} = {:.3f}'.format(args.metric, ensemble_score))

    return ensemble_score


def cross_validate(args: Namespace):
    """k-fold cross validation"""
    init_seed = args.seed
    save_dir = args.save_dir

    # Run training on different random seeds for each fold
    test_scores = []
    for fold_num in range(args.num_folds):
        logger.info('Fold {}'.format(fold_num))
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, 'fold_{}'.format(fold_num))
        os.makedirs(args.save_dir, exist_ok=True)
        test_scores.append(run_training(args))

    # Report results
    logger.info('{}-fold cross validation'.format(args.num_folds))
    for fold_num, score in enumerate(test_scores):
        logger.info('Seed {} ==> test {} = {:.3f}'.format(init_seed + fold_num, args.metric, score))
    logger.info('Overall test {} = {:.3f} ± {:.3f}'.format(args.metric, np.mean(test_scores), np.std(test_scores)))


if __name__ == '__main__':
    args = parse_args()
    set_logger(logger, args.save_dir, args.quiet)
    cross_validate(args)
