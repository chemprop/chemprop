from argparse import Namespace
from logging import Logger
import os
from pprint import pformat
from typing import List

import numpy as np
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
import pickle
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.data import cluster_split, generate_unsupervised_cluster_labels, MoleculeDataset, StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_desired_labels, get_task_names, split_data, \
    truncate_outliers, load_prespecified_chunks
from chemprop.models import build_model
from chemprop.nn_utils import param_count, compute_pnorm
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
    args.features_size = data.features_size()
    args.real_num_tasks = args.num_tasks - args.features_size if args.predict_features else args.num_tasks
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
            test_data = get_data(args.separate_test_set, args)
            if args.separate_val_set:
                val_data = get_data(args.separate_val_set, args)
                train_data = data  # nothing to split; we already got our test and val sets
            else:
                train_data, val_data, _ = split_data(data, args, sizes=(0.8, 0.2, 0.0), seed=args.seed, logger=logger)
        else:
            train_data, val_data, test_data = split_data(data, args, sizes=args.split_sizes, seed=args.seed, logger=logger)

    # Optionally replace test data with train or val data
    if args.test_split == 'train':
        test_data = train_data
    elif args.test_split == 'val':
        test_data = val_data

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug('{} '.format(args.task_names[i]) +
                  ', '.join('{}: {:.2f}%'.format(cls, size * 100) for cls, size in enumerate(task_class_sizes)))

        if args.class_balance:
            train_class_sizes = get_class_sizes(train_data)
            class_batch_counts = torch.Tensor(train_class_sizes) * args.batch_size
            args.class_weights = 1 / torch.Tensor(class_batch_counts)

    if args.save_smiles_splits:
        with open(args.data_path, 'r') as f:
            header = f.readline().strip()
            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(f):
                line = line.strip()
                smiles = line.split(',')[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i

        all_split_indices = []
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
                f.write('smiles\n')
                for smiles in dataset.smiles():
                    f.write(smiles.strip() + '\n')
            with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
                f.write(header + '\n')
                for smiles in dataset.smiles():
                    f.write(lines_by_smiles[smiles] + '\n')
            split_indices = []
            for smiles in dataset.smiles():
                split_indices.append(indices_by_smiles[smiles])
                split_indices = sorted(split_indices)
            all_split_indices.append(split_indices)
        with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=None if args.predict_features else 0)
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
        train_data.set_targets(scaled_targets)
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
    if args.maml:  # TODO refactor
        test_targets = []
        for task_idx in range(len(data.data[0].targets)):
            _, task_test_data, _ = test_data.sample_maml_task(args, seed=0)
            test_targets += task_test_data.targets()

    if args.dataset_type == 'bert_pretraining':
        sum_test_preds = {
            'features': np.zeros((len(test_smiles), args.features_size)) if args.features_size is not None else None,
            'vocab': np.zeros((len(test_targets['vocab']), args.vocab.output_size))
        }
    elif args.dataset_type == 'kernel':
        sum_test_preds = np.zeros((len(test_targets), args.num_tasks))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    if args.maml:
        sum_test_preds = None  # annoying to determine exact size; will initialize later

    if args.dataset_type == 'bert_pretraining':
        # Only predict targets that are masked out
        test_targets['vocab'] = [target if mask == 0 else None for target, mask in zip(test_targets['vocab'], test_data.mask())]

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

        if args.adjust_weight_decay:
            args.pnorm_target = compute_pnorm(model)

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
            
            if args.gradual_unfreezing:
                if epoch % args.epochs_per_unfreeze == 0:
                    unfroze_layer = model.unfreeze_next()  # consider just stopping early after we have nothing left to unfreeze?
                    if unfroze_layer:
                        debug('Unfroze last frozen layer')

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
                scaler=scaler,
                logger=logger
            )

            if args.dataset_type == 'bert_pretraining':
                if val_scores['features'] is not None:
                    debug('Validation features rmse = {:.3f}'.format(val_scores['features']))
                    writer.add_scalar('validation_features_rmse', val_scores['features'], n_iter)
                val_scores = [val_scores['vocab']]

            # Average validation score
            avg_val_score = np.nanmean(val_scores)
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

        if args.split_test_by_overlap_dataset is not None:
            overlap_data = get_data(args.split_test_by_overlap_dataset)
            overlap_smiles = set(overlap_data.smiles())
            test_data_intersect, test_data_nonintersect = [], []
            for d in test_data.data:
                if d.smiles in overlap_smiles:
                    test_data_intersect.append(d)
                else:
                    test_data_nonintersect.append(d)
            test_data_intersect, test_data_nonintersect = MoleculeDataset(test_data_intersect), MoleculeDataset(test_data_nonintersect)
            for name, td in [('Intersect', test_data_intersect), ('Nonintersect', test_data_nonintersect)]:
                test_preds = predict(
                    model=model,
                    data=td,
                    args=args,
                    scaler=scaler,
                    logger=logger
                )
                test_scores = evaluate_predictions(
                    preds=test_preds,
                    targets=td.targets(),
                    metric_func=metric_func,
                    args=args,
                    logger=logger
                )
                avg_test_score = np.nanmean(test_scores)
                info('Model {} test {} for {} = {:.3f}'.format(model_idx, args.metric, name, avg_test_score))
        
        if len(test_data) == 0:  # just get some garbage results without crashing; in this case we didn't care anyway
            test_preds, test_scores = sum_test_preds, [0 for _ in range(len(args.task_names))]
        else:
            test_preds = predict(
                model=model,
                data=test_data,
                args=args,
                scaler=scaler,
                logger=logger
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                metric_func=metric_func,
                args=args,
                logger=logger
            )

        if args.maml:
            if sum_test_preds is None:
                sum_test_preds = np.zeros(np.array(test_preds).shape)

        if args.dataset_type == 'bert_pretraining':
            if test_preds['features'] is not None:
                sum_test_preds['features'] += np.array(test_preds['features'])
            sum_test_preds['vocab'] += np.array(test_preds['vocab'])
        else:
            sum_test_preds += np.array(test_preds)

        if args.dataset_type == 'bert_pretraining':
            if test_preds['features'] is not None:
                debug('Model {} test features rmse = {:.3f}'.format(model_idx, test_scores['features']))
                writer.add_scalar('test_features_rmse', test_scores['features'], 0)
            test_scores = [test_scores['vocab']]

        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info('Model {} test {} = {:.3f}'.format(model_idx, args.metric, avg_test_score))
        writer.add_scalar('test_{}'.format(args.metric), avg_test_score, 0)

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                if task_name in desired_labels:
                    info('Model {} test {} {} = {:.3f}'.format(model_idx, task_name, args.metric, test_score))
                    writer.add_scalar('test_{}_{}'.format(task_name, args.metric), test_score, n_iter)

    # Evaluate ensemble on test set
    if args.dataset_type == 'bert_pretraining':
        avg_test_preds = {
            'features': (sum_test_preds['features'] / args.ensemble_size).tolist() if sum_test_preds['features'] is not None else None,
            'vocab': (sum_test_preds['vocab'] / args.ensemble_size).tolist()
        }
    else:
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

    if len(test_data) == 0:  # just return some garbage when we didn't want test data
        ensemble_scores = test_scores
    else:
        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            metric_func=metric_func, 
            args=args,
            logger=logger
        )

    # Average ensemble score
    if args.dataset_type == 'bert_pretraining':
        if ensemble_scores['features'] is not None:
            info('Ensemble test features rmse = {:.3f}'.format(ensemble_scores['features']))
            writer.add_scalar('ensemble_test_features_rmse', ensemble_scores['features'], 0)
        ensemble_scores = [ensemble_scores['vocab']]

    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    info('Ensemble test {} = {:.3f}'.format(args.metric, avg_ensemble_test_score))
    writer.add_scalar('ensemble_test_{}'.format(args.metric), avg_ensemble_test_score, 0)

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info('Ensemble test {} {} = {:.3f}'.format(task_name, args.metric, ensemble_score))

    return ensemble_scores
