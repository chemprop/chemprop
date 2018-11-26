from argparse import Namespace
from logging import Logger
import os
from pprint import pformat
from typing import List

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import trange
import pickle
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.data import cluster_split, generate_unsupervised_cluster_labels, MoleculeDataset, StandardScaler
from chemprop.data.utils import get_data, get_desired_labels, get_task_names, split_data, truncate_outliers,\
    load_prespecified_chunks
from chemprop.models import build_model
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    save_checkpoint


def run_training(args: Namespace, logger: Logger = None) -> List[float]:
    """Trains a model and returns test scores on the model checkpoint with the highest validation score"""
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print
    
    debug(pformat(vars(args)))

    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    desired_labels = get_desired_labels(args, args.task_names)
    data = get_data(args.data_path, args)
    args.num_tasks = data.num_tasks()
    debug('Number of tasks = {}'.format(args.num_tasks))

    if args.dataset_type == 'bert_pretraining':
        data.bert_init(args, logger)

    # Split data
    if args.dataset_type == 'regression_with_binning':  # Note: for now, binning based on whole dataset, not just training set
        data, bin_predictions, regression_data = data
        args.bin_predictions = bin_predictions
        debug('Splitting data with seed {}'.format(args.seed))
        train_data, _, _ = split_data(data, args, sizes=args.split_sizes, seed=args.seed, logger=logger)
        _, val_data, test_data = split_data(regression_data, args, sizes=args.split_sizes, seed=args.seed, logger=logger)
    else:
        debug('Splitting data with seed {}'.format(args.seed))
        if args.separate_test_set:
            train_data, val_data, _ = split_data(data, args, sizes=(0.8, 0.2, 0.0), seed=args.seed, logger=logger)
            test_data = get_data(args.separate_test_set, args) 
        else:
            train_data, val_data, test_data = split_data(data, args, sizes=args.split_sizes, seed=args.seed, logger=logger)

    if args.features_scaling:
        features_scaler = train_data.normalize_features()
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data) if args.prespecified_chunk_dir is None else args.prespecified_chunks_max_examples_per_epoch

    if args.adversarial or args.moe:
        val_smiles, test_smiles = val_data.smiles(), test_data.smiles()

    debug('Total size = {:,} | train size = {:,} | val size = {:,} | test size = {:,}'.format(
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
    if args.dataset_type == 'regression' and args.target_scaling:
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        for i in range(len(train_data)):
            train_data[i].targets = scaled_targets[i]
    else:
        scaler = None

    if args.moe:
        train_data = cluster_split(train_data, 
                                   args.num_sources, 
                                   args.cluster_max_ratio, 
                                   seed=args.cluster_split_seed, 
                                   logger=logger)

    # Chunk training data if too large to load in memory all at once
    if args.num_chunks > 1:
        os.makedirs(args.chunk_temp_dir, exist_ok=True)
        train_paths = []
        if args.moe:
            chunked_sources = [td.chunk(args.num_chunks) for td in train_data]
            chunks = []
            for i in range(args.num_chunks):
                chunks.append([source[i] for source in chunked_sources])
        else:
            chunks = train_data.chunk(args.num_chunks)
        for i in range(args.num_chunks):
            chunk_path = os.path.join(args.chunk_temp_dir, str(i) + '.txt')
            memo_path = os.path.join(args.chunk_temp_dir, 'memo' + str(i) + '.txt')
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunks[i], f)
            train_paths.append((chunk_path, memo_path))
        train_data = train_paths

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(args)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()

    if args.dataset_type == 'bert_pretraining':
        sum_test_preds = np.zeros((len(test_targets), args.vocab.output_size))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    if args.dataset_type == 'bert_pretraining':
        # Only predict targets that are masked out
        test_targets = [target if mask == 0 else None for target, mask in zip(test_targets, test_data.mask())]

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, 'model_{}'.format(model_idx))
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug('Loading model {} from {}'.format(model_idx, args.checkpoint_paths[model_idx]))
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug('Building model {}'.format(model_idx))
            model = build_model(args)

        debug(model)
        debug('Number of parameters = {:,}'.format(param_count(model)))
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(model, scaler, features_scaler, args, os.path.join(save_dir, 'model.pt'))

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug('Epoch {}'.format(epoch))

            if args.prespecified_chunk_dir is not None:
                # load some different random chunks each epoch
                train_data, val_data = load_prespecified_chunks(args, logger)
                debug('Loaded prespecified chunks for epoch')

            if args.dataset_type == 'unsupervised':  # won't work with moe
                full_data = MoleculeDataset(train_data.data + val_data.data)
                generate_unsupervised_cluster_labels(build_model(args), full_data, args)  # cluster with a new random init
                model.create_ffn(args)  # reset the ffn since we're changing targets-- we're just pretraining the encoder.
                optimizer.param_groups.pop()  # remove ffn parameters
                optimizer.add_param_group({'params': model.ffn.parameters(), 'lr': args.init_lr[1], 'weight_decay': args.weight_decay[1]})
                if args.cuda:
                    model.ffn.cuda()

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
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data=val_data,
                metric_func=metric_func,
                args=args,
                scaler=scaler
            )

            # Average validation score
            avg_val_score = np.mean(val_scores)
            debug('Validation {} = {:.3f}'.format(args.metric, avg_val_score))
            writer.add_scalar('validation_{}'.format(args.metric), avg_val_score, n_iter)

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    if task_name in desired_labels:
                        debug('Validation {} {} = {:.3f}'.format(task_name, args.metric, val_score))
                        writer.add_scalar('validation_{}_{}'.format(task_name, args.metric), val_score, n_iter)

            # Save model checkpoint if improved validation score, or always save it if unsupervised
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score or \
                    args.dataset_type == 'unsupervised':
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(model, scaler, features_scaler, args, os.path.join(save_dir, 'model.pt'))

        if args.dataset_type == 'unsupervised':
            return [0]  # rest of this is meaningless when unsupervised

        # Evaluate on test set using model with best validation score
        info('Model {} best validation {} = {:.3f} on epoch {}'.format(model_idx, args.metric, best_score, best_epoch))
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)
        test_preds = predict(
            model=model,
            data=test_data,
            args=args,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            metric_func=metric_func,
            args=args
        )
        sum_test_preds += np.array(test_preds)

        # Average test score
        avg_test_score = np.mean(test_scores)
        info('Model {} test {} = {:.3f}'.format(model_idx, args.metric, avg_test_score))
        writer.add_scalar('test_{}'.format(args.metric), avg_test_score, n_iter)

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                if task_name in desired_labels:
                    info('Model {} test {} {} = {:.3f}'.format(model_idx, task_name, args.metric, test_score))
                    writer.add_scalar('test_{}_{}'.format(task_name, args.metric), test_score, n_iter)

    # Evaluate ensemble on test set
    avg_test_preds = sum_test_preds / args.ensemble_size
    ensemble_scores = evaluate_predictions(
        preds=list(avg_test_preds),
        targets=test_targets,
        metric_func=metric_func, 
        args=args
    )

    # Average ensemble score
    info('Ensemble test {} = {:.3f}'.format(args.metric, np.mean(ensemble_scores)))

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info('Ensemble test {} {} = {:.3f}'.format(task_name, args.metric, ensemble_score))

    return ensemble_scores
