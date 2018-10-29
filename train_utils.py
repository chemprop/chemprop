from argparse import Namespace
import logging
import random
from typing import Callable, List, Union
import os

from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from tqdm import trange, tqdm
import numpy as np
import pickle

from data import MoleculeDataset
from nn_utils import NoamLR
from utils import compute_gnorm, compute_pnorm
import featurization


def train(model: nn.Module,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: NoamLR,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None,
          chunk_names: bool = False,
          val_smiles: List[str] = None,
          test_smiles: List[str] = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: Optimizer.
    :param scheduler: A NoamLR learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :param chunk_names: Whether to train on the data in chunks. In this case,
    data must be a list of paths to the data chunks.
    :param val_smiles: Validation smiles strings without targets.
    :param test_smiles: Test smiles strings without targets, used for adversarial setting.
    :return: The total number of iterations (training examples) trained on so far.
    """
    model.train()
    
    if not args.moe:
        data.shuffle()

    if chunk_names:
        for path, memo_path in tqdm(data, total=len(data)):
            featurization.SMILES_TO_FEATURES = dict()
            if os.path.isfile(memo_path):
                found_memo = True
                with open(memo_path, 'rb') as f:
                    featurization.SMILES_TO_FEATURES = pickle.load(f)
            else:
                found_memo = False
            with open(path, 'rb') as f:
                chunk = pickle.load(f)
            random.shuffle(chunk)
            n_iter = train(
                model=model,
                data=chunk,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer,
                chunk_names=False,
                val_smiles=val_smiles,
                test_smiles=test_smiles
            )
            if not found_memo:
                with open(memo_path, 'wb') as f:
                    pickle.dump(featurization.SMILES_TO_GRAPH, f, protocol=pickle.HIGHEST_PROTOCOL)
        return n_iter

    loss_sum, iter_count = 0, 0
    if args.adversarial:
        train_smiles = data.smiles()
        train_val_smiles = train_smiles + val_smiles
        d_loss_sum, g_loss_sum, gp_norm_sum = 0, 0, 0

    if args.moe:
        test_smiles = list(test_smiles)
        random.shuffle(test_smiles)
        train_smiles = []
        for d in data:
            d.shuffle()
            train_smiles.append(d.smiles())
        num_iters = min(len(test_smiles), min([len(d) for d in data]))
    else:
        num_iters = len(data)

    for i in trange(0, num_iters, args.batch_size):
        if args.moe:
            model.compute_domain_encs(train_smiles) # want to recompute every batch TODO(moe) change this if slow
            batch = [MoleculeDataset(d[i:i + args.batch_size]) for d in data]
            train_batch, train_targets = [], []
            for b in batch:
                tb, tt = b.smiles(), b.targets()
                train_batch.append(tb)
                train_targets.append(tt)
            test_batch = test_smiles[i:i + args.batch_size]
            loss = model.compute_loss(train_batch, train_targets, test_batch)
            model.zero_grad()
            if logger is not None:
                loss_sum += loss.item()
                iter_count += len(batch)
        else:
            # Prepare batch
            batch = MoleculeDataset(data[i:i + args.batch_size])
            smiles_batch, features_batch, target_batch = batch.smiles(), batch.features(), batch.targets()

            mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
            targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

            if next(model.parameters()).is_cuda:
                mask, targets = mask.cuda(), targets.cuda()

            # Run model
            model.zero_grad()
            preds = model(smiles_batch, features_batch)
            if args.dataset_type == 'regression_with_binning':
                preds = preds.view(targets.size(0), targets.size(1), -1)
                targets = targets.long()
                loss = 0
                for task in range(targets.size(1)):
                    loss += loss_func(preds[:, task, :], targets[:, task]) * mask[:, task]  # for some reason cross entropy doesn't support multi target
            else:
                loss = loss_func(preds, targets) * mask
            loss = loss.sum() / mask.sum()

            if logger is not None:
                loss_sum += loss.item()
                iter_count += len(batch)

        loss.backward()
        if args.max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if args.adversarial:
            for _ in range(args.gan_d_per_g):
                train_val_smiles_batch = random.sample(train_val_smiles, args.batch_size)
                test_smiles_batch = random.sample(test_smiles, args.batch_size)
                d_loss, gp_norm = model.train_D(train_val_smiles_batch, test_smiles_batch)
            train_val_smiles_batch = random.sample(train_val_smiles, args.batch_size)
            test_smiles_batch = random.sample(test_smiles, args.batch_size)
            g_loss = model.train_G(train_val_smiles_batch, test_smiles_batch)
            if logger is not None:
                # we probably only care about the g_loss honestly
                d_loss_sum += d_loss * args.batch_size
                gp_norm_sum += gp_norm * args.batch_size
                g_loss_sum += g_loss * args.batch_size

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0 and (logger is not None or writer is not None):
            lr = scheduler.get_lr()[0]
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            if args.adversarial:
                d_loss_avg, g_loss_avg, gp_norm_avg = d_loss_sum / iter_count, g_loss_sum / iter_count, gp_norm_sum / iter_count
                d_loss_sum, g_loss_sum, gp_norm_sum = 0, 0, 0
            loss_sum, iter_count = 0, 0

            if logger is not None:
                logger.debug("Loss = {:.4e}, PNorm = {:.4f}, GNorm = {:.4f}, lr = {:.4e}".format(loss_avg, pnorm, gnorm, lr))
                if args.adversarial:
                    logger.debug("D Loss = {:.4e}, G Loss = {:.4e}, GP Norm = {:.4}".format(d_loss_avg, g_loss_avg, gp_norm_avg))

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                writer.add_scalar('learning_rate', lr, n_iter)

    return n_iter


def predict(model: nn.Module,
            data: MoleculeDataset,
            args: Namespace,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param args: Arguments.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    with torch.no_grad():
        model.eval()

        preds = []
        for i in range(0, len(data), args.batch_size):
            # Prepare batch
            batch = MoleculeDataset(data[i:i + args.batch_size])
            smiles_batch, features_batch = batch.smiles(), batch.features()

            # Run model
            batch_preds = model(smiles_batch, features_batch)
            batch_preds = batch_preds.data.cpu().numpy()
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
            
            if args.dataset_type == 'regression_with_binning':
                batch_preds = batch_preds.reshape((batch_preds.shape[0], args.num_tasks, args.num_bins))
                indices = np.argmax(batch_preds, axis=2)
                preds.extend(indices.tolist())
            else:
                preds.extend(batch_preds.tolist())
        
        if args.dataset_type == 'regression_with_binning':
            preds = args.bin_predictions[np.array(preds)].tolist()

        return preds


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         metric_func: Callable) -> List[float]:
    """
    Evaluates predictions using a metric function and filtering out invalid targets.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param targets: A list of lists of shape (data_size, num_tasks) with targets.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :return: A list with the score for each task based on `metric_func`.
    """
    data_size, num_tasks = len(preds), len(preds[0])

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(data_size):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = []
    for i in range(num_tasks):
        # Skip if all targets are identical
        if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
            continue
        results.append(metric_func(valid_targets[i], valid_preds[i]))

    return results


def evaluate(model: nn.Module,
             data: MoleculeDataset,
             metric_func: Callable,
             args: Namespace,
             scaler: StandardScaler = None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param args: Arguments.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list with the score for each task based on `metric_func`.
    """
    smiles, targets = data.smiles(), data.targets()

    preds = predict(
        model=model,
        data=data,
        args=args,
        scaler=scaler
    )

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        metric_func=metric_func
    )

    return results
