from argparse import Namespace
import logging
import os
import math
from pprint import pformat
from typing import List

import numpy as np
from tensorboardX import SummaryWriter
from torch.optim import Adam
from tqdm import trange
import pickle
import shutil

from model import build_model
from nn_utils import NoamLR, param_count
from parsing import parse_train_args
from train_utils import train, predict, evaluate, evaluate_predictions
from utils import get_data, get_task_names, get_desired_labels, get_loss_func, get_metric_func, load_checkpoint, \
    save_checkpoint, set_logger, split_data, truncate_outliers, StandardScaler
from scaffold import cluster_split


# Initialize logger
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)


def run_training(args: Namespace) -> List[float]:
    """Trains a model and returns test scores on the model checkpoint with the highest validation score"""
    logger.debug(pformat(vars(args)))

    logger.debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    args.num_tasks = len(args.task_names)
    logger.debug('Number of tasks = {}'.format(args.num_tasks))
    desired_labels = get_desired_labels(args, args.task_names)
    data = get_data(args.data_path, args)

    if args.dataset_type == 'regression_with_binning':  # Note: for now, binning based on whole dataset, not just training set
        data, bin_predictions, regression_data = data
        args.bin_predictions = bin_predictions
        logger.debug('Splitting data with seed {}'.format(args.seed))
        train_data, _, _ = split_data(data, args, sizes=args.split_sizes, seed=args.seed, logger=logger)
        _, val_data, test_data = split_data(regression_data, args, sizes=args.split_sizes, seed=args.seed, logger=logger)
    else:
        logger.debug('Splitting data with seed {}'.format(args.seed))
        if args.separate_test_set:
            train_data, val_data, _ = split_data(data, args, sizes=(0.8, 0.2, 0.0), seed=args.seed, logger=logger)
            test_data = get_data(args.separate_test_set, args) 
        else:
            train_data, val_data, test_data = split_data(data, args, sizes=args.split_sizes, seed=args.seed, logger=logger)
    
    if args.adversarial or args.moe:
        val_smiles, test_smiles = val_data.smiles(), test_data.smiles()
        args.train_data_length = len(train_data)  # kinda hacky, but less cluttered

    logger.debug('Total size = {:,} | train size = {:,} | val size = {:,} | test size = {:,}'.format(
        len(data),
        len(train_data),
        len(val_data),
        len(test_data))
    )

    # Optionally truncate outlier values
    if args.truncate_outliers:
        print('Truncating outliers in train set')
        train_data = truncate_outliers(train_data)

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        logger.debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        for i in range(len(train_data)):
            train_data[i].targets = scaled_targets[i]
    else:
        scaler = None

    # Chunk training data if too large to load in memory all at once
    train_data_length = len(train_data)
    if args.num_chunks > 1:
        chunk_len = math.ceil(len(train_data) / args.num_chunks)
        os.makedirs(args.chunk_temp_dir, exist_ok=True)
        train_paths = []
        for i in range(args.num_chunks):
            chunk_path = os.path.join(args.chunk_temp_dir, str(i) + '.txt')
            memo_path = os.path.join(args.chunk_temp_dir, 'memo' + str(i) + '.txt')
            with open(chunk_path, 'wb') as f:
                pickle.dump(train_data[i * chunk_len:(i + 1) * chunk_len], f)
            train_paths.append((chunk_path, memo_path))
        train_data = train_paths
    
    if args.moe:
        train_data = cluster_split(train_data, args.num_sources)

    # Get loss and metric functions
    loss_func = get_loss_func(args.dataset_type)
    metric_func = get_metric_func(args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, 'model_{}'.format(model_idx))
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            logger.debug('Loading model {} from {}'.format(model_idx, args.checkpoint_paths[model_idx]))
            model = load_checkpoint(args.checkpoint_paths[model_idx])
        else:
            logger.debug('Building model {}'.format(model_idx))
            model = build_model(args)

        logger.debug(model)
        logger.debug('Number of parameters = {:,}'.format(param_count(model)))
        if args.cuda:
            logger.debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(model, scaler, args, os.path.join(save_dir, 'model.pt'))

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
                chunk_names=(args.num_chunks > 1),
                val_smiles=val_smiles if args.adversarial else None,
                test_smiles=test_smiles if args.adversarial or args.moe else None
            )
            val_scores = evaluate(
                model=model,
                data=val_data,
                metric_func=metric_func,
                args=args,
                scaler=scaler
            )

            # Average validation score
            avg_val_score = np.mean(val_scores)
            logger.debug('Validation {} = {:.3f}'.format(args.metric, avg_val_score))
            writer.add_scalar('validation_{}'.format(args.metric), avg_val_score, n_iter)

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    if task_name in desired_labels:
                        logger.debug('Validation {} {} = {:.3f}'.format(task_name, args.metric, val_score))
                        writer.add_scalar('validation_{}_{}'.format(task_name, args.metric), val_score, n_iter)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(model, scaler, args, os.path.join(save_dir, 'model.pt'))

        # Evaluate on test set using model using model with best validation score
        logger.info('Model {} best validation {} = {:.3f} on epoch {}'.format(model_idx, args.metric, best_score, best_epoch))
        model = load_checkpoint(os.path.join(args.save_dir, 'model_{}/model.pt'.format(model_idx)), cuda=args.cuda)
        test_preds = predict(
            model=model,
            data=test_data,
            args=args,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            metric_func=metric_func
        )
        sum_test_preds += np.array(test_preds)

        # Average test score
        avg_test_score = np.mean(test_scores)
        logger.info('Model {} test {} = {:.3f}'.format(model_idx, args.metric, avg_test_score))
        writer.add_scalar('test_{}'.format(args.metric), avg_test_score, n_iter)

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                if task_name in desired_labels:
                    logger.info('Model {} test {} {} = {:.3f}'.format(model_idx, task_name, args.metric, test_score))
                    writer.add_scalar('test_{}_{}'.format(task_name, args.metric), test_score, n_iter)

    # Evaluate ensemble on test set
    avg_test_preds = sum_test_preds / args.ensemble_size
    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds.tolist(),
        targets=test_targets,
        metric_func=metric_func
    )

    # Average ensemble score
    logger.info('Ensemble test {} = {:.3f}'.format(args.metric, np.mean(ensemble_scores)))

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            logger.info('Ensemble test {} {} = {:.3f}'.format(task_name, args.metric, ensemble_score))

    return ensemble_scores


def cross_validate(args: Namespace):
    """k-fold cross validation"""
    set_logger(logger, args.save_dir, args.quiet)

    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)
    desired_labels = get_desired_labels(args, task_names)

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        logger.info('Fold {}'.format(fold_num))
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, 'fold_{}'.format(fold_num))
        os.makedirs(args.save_dir, exist_ok=True)
        model_scores = run_training(args)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report results
    logger.info('{}-fold cross validation'.format(args.num_folds))

    # Report scores for each model
    for fold_num, scores in enumerate(all_scores):
        logger.info('Seed {} ==> test {} = {:.3f}'.format(init_seed + fold_num, args.metric, np.mean(scores)))

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                if task_name in desired_labels:
                    logger.info('Seed {} ==> test {} {} = {:.3f}'.format(init_seed + fold_num, task_name, args.metric, score))

    # Report scores across models
    avg_scores = np.mean(all_scores, axis=1)  # average score for each model across tasks
    logger.info('Overall test {} = {:.3f} ± {:.3f}'.format(args.metric, np.mean(avg_scores), np.std(avg_scores)))

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            if task_name in desired_labels:
                logger.info('Overall test {} {} = {:.3f} +/- {:.3f}'.format(
                    task_name,
                    args.metric,
                    np.mean(all_scores[:, task_num]),
                    np.std(all_scores[:, task_num]))
                )

    if args.num_chunks > 1:
        shutil.rmtree(args.chunk_temp_dir)


if __name__ == '__main__':
    args = parse_train_args()
    cross_validate(args)
