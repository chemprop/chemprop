from argparse import Namespace
import logging
import os
import math
from pprint import pformat

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.optim import Adam
from tqdm import trange
import pickle

from model import build_model
from nn_utils import NoamLR, param_count
from parsing import parse_args
from train_utils import train, predict, evaluate, evaluate_predictions
from utils import get_data, get_loss_func, get_metric_func, set_logger, split_data, truncate_outliers, StandardScaler


# Initialize logger
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)


def run_training(args: Namespace) -> float:
    """Trains a model and returns test score on the model checkpoint with the highest validation score"""
    logger.debug(pformat(vars(args)))

    logger.debug('Loading data')
    data = get_data(args.data_path, args.dataset_type, num_bins=args.num_bins) 
    if args.dataset_type == 'regression_with_binning': #note: for now, binning based on whole dataset, not just training set
        data, bin_predictions, regression_data = data
        args.bin_predictions = bin_predictions
        logger.debug('Splitting data with seed {}'.format(args.seed))
        train_data, _, _ = split_data(data, seed=args.seed)
        _, val_data, test_data = split_data(regression_data, seed=args.seed)
    else:
        logger.debug('Splitting data with seed {}'.format(args.seed))
        if args.separate_test_set:
            train_data, val_data = split_data(data, sizes=(0.8, 0.2, 0), seed=args.seed)
            test_data = get_data(args.separate_test_set, args.dataset_type, num_bins=args.num_bins) 
        else:
            train_data, val_data, test_data = split_data(data, seed=args.seed)
    num_tasks = len(data[0][1])
    args.num_tasks = num_tasks

    logger.debug('Train size = {:,} | val size = {:,} | test size = {:,}'.format(
        len(train_data),
        len(val_data),
        len(test_data))
    )
    logger.debug('Number of tasks = {}'.format(num_tasks))

    # Optionally truncate outlier values
    if args.truncate_outliers:
        print('Truncating outliers in train set')
        train_data = truncate_outliers(train_data)

    # Initialize scaler and scaler training labels by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        logger.debug('Fitting scaler')
        train_smiles, train_labels = zip(*train_data)
        scaler = StandardScaler().fit(train_labels)
        train_data = list(zip(train_smiles, scaler.transform(train_labels).tolist()))
    else:
        scaler = None

    train_data_length = len(train_data)
    if args.num_chunks > 1:
        chunk_len = math.ceil(len(train_data) / args.num_chunks)
        os.makedirs(args.chunk_temp_dir, exist_ok=True)
        train_paths = []
        for i in range(args.num_chunks):
            chunk_path = os.path.join(args.chunk_temp_dir, str(i) + '.txt')
            with open(chunk_path, 'wb') as f:
                pickle.dump(train_data[i*chunk_len:(i+1)*chunk_len], f)
            train_paths.append(chunk_path)
        train_data = train_paths

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
        model = build_model(num_tasks, args)

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
        optimizer = Adam(model.parameters(), lr=args.init_lr)
        scheduler = NoamLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            steps_per_epoch=train_data_length // args.batch_size,
            init_lr=args.init_lr,
            max_lr=args.max_lr,
            final_lr=args.final_lr
        )

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            logger.debug('Epoch {}'.format(epoch))

            n_iter = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer,
                chunk_names=(args.num_chunks>1)
            )
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
        model_score, _ = evaluate_predictions(
            preds=model_preds,
            labels=labels,
            metric_func=metric_func
        )
        logger.info('Model {} test {} = {:.3f}'.format(model_idx, args.metric, model_score))

        sum_preds += np.array(model_preds)

    # Evaluate ensemble
    avg_preds = sum_preds / args.ensemble_size
    ensemble_score, ensemble_indiv_scores = evaluate_predictions(
        preds=avg_preds.tolist(),
        labels=labels,
        metric_func=metric_func
    )
    logger.info('Ensemble test {} = {:.3f}'.format(args.metric, ensemble_score))

    return ensemble_score, ensemble_indiv_scores


def cross_validate(args: Namespace):
    """k-fold cross validation"""
    init_seed = args.seed
    save_dir = args.save_dir

    # Run training on different random seeds for each fold
    test_scores = []
    indiv_test_scores = []
    for fold_num in range(args.num_folds):
        logger.info('Fold {}'.format(fold_num))
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, 'fold_{}'.format(fold_num))
        os.makedirs(args.save_dir, exist_ok=True)
        ensemble_scores, ensemble_indiv_scores = run_training(args)
        test_scores.append(ensemble_scores)
        indiv_test_scores.append(ensemble_indiv_scores)

    # Report results
    logger.info('{}-fold cross validation'.format(args.num_folds))
    for fold_num, score in enumerate(test_scores):
        logger.info('Seed {} ==> test {} = {:.3f}'.format(init_seed + fold_num, args.metric, score))
    logger.info('Overall test {} = {:.3f} ± {:.3f}'.format(args.metric, np.mean(test_scores), np.std(test_scores)))
    if args.show_individual_scores:
        logger.info('Individual task scores: {}'.format(np.mean(indiv_test_scores, axis=0)))
    if args.num_chunks > 1:
        os.removedirs(args.chunk_temp_dir)

if __name__ == '__main__':
    args = parse_args()
    set_logger(logger, args.save_dir, args.quiet)
    cross_validate(args)
