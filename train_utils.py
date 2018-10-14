from argparse import Namespace
import logging
import random
from typing import Callable, List, Tuple
import os

from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from tqdm import trange
import numpy as np
import pickle
from tqdm import tqdm

from nn_utils import NoamLR
from utils import compute_gnorm, compute_pnorm
import featurization

def train(model: nn.Module,
          data: List[Tuple[str, List[float]]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: NoamLR,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None,
          chunk_names: bool = False) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: Training data.
    :param loss_func: Loss function.
    :param optimizer: Optimizer.
    :param scheduler: A NoamLR learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :param chunk_names: Whether to train on the data in chunks. In this case,
    data must be a list of paths to the data chunks.
    :return: The total number of iterations (training examples) trained on so far.
    """
    model.train()
    random.shuffle(data)

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
            n_iter = train( model=model, 
                            data=chunk, 
                            loss_func=loss_func, 
                            optimizer=optimizer, 
                            scheduler=scheduler, 
                            args=args, 
                            n_iter=n_iter,
                            logger=logger,
                            writer=writer,
                            chunk_names=False)
            if not found_memo:
                with open(memo_path, 'wb') as f:
                    pickle.dump(featurization.SMILES_TO_FEATURES, f, protocol=pickle.HIGHEST_PROTOCOL)
        return n_iter

    loss_sum, iter_count = 0, 0
    for i in trange(0, len(data), args.batch_size):
        # Prepare batch
        batch = data[i:i + args.batch_size]
        mol_batch, label_batch = zip(*batch)

        mask = torch.Tensor([[x is not None for x in lb] for lb in label_batch])
        labels = torch.Tensor([[0 if x is None else x for x in lb] for lb in label_batch])

        if next(model.parameters()).is_cuda:
            mask, labels = mask.cuda(), labels.cuda()

        # Run model
        model.zero_grad()
        preds = model(mol_batch)
        if args.dataset_type == 'regression_with_binning':
            preds = preds.view(labels.size(0), labels.size(1), -1)
            labels = labels.long()
            loss = 0
            for task in range(labels.size(1)):
                loss += loss_func(preds[:, task, :], labels[:, task]) * mask[:, task]  # for some reason cross entropy doesn't support multi target
        else:
            loss = loss_func(preds, labels) * mask
        loss = loss.sum() / mask.sum()

        if logger is not None:
            loss_sum += loss.item()
            iter_count += len(batch)

        loss.backward()
        if args.max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0 and (logger is not None or writer is not None):
            lr = scheduler.get_lr()[0]
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            if logger is not None:
                logger.debug("Loss = {:.4e}, PNorm = {:.4f}, GNorm = {:.4f}, lr = {:.4f}".format(loss_avg, pnorm, gnorm, lr))

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                writer.add_scalar('learning_rate', lr, n_iter)

    return n_iter


def predict(model: nn.Module,
            smiles: List[str],
            args: Namespace,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param smiles: A list of smiles strings.
    :param args: Arguments.
    :param scaler: A StandardScaler object fit on the training labels.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    with torch.no_grad():
        model.eval()

        preds = []
        for i in range(0, len(smiles), args.batch_size):
            # Prepare batch
            mol_batch = smiles[i:i + args.batch_size]

            # Run model
            batch_preds = model(mol_batch)
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
                         labels: List[List[float]],
                         metric_func: Callable) -> List[float]:
    """
    Evaluates predictions using a metric function and filtering out invalid labels.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param labels: A list of lists of shape (data_size, num_tasks) with labels.
    :param metric_func: Metric function which takes in a list of labels and a list of predictions.
    :return: A list with the score for each task based on `metric_func`.
    """
    data_size, num_tasks = len(preds), len(preds[0])

    # Filter out empty labels
    # valid_preds and valid_labels have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_labels = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(data_size):
            if labels[j][i] is not None:  # Skip those without labels
                valid_preds[i].append(preds[j][i])
                valid_labels[i].append(labels[j][i])

    # Compute metric
    results = []
    for i in range(num_tasks):
        # Skip if all labels are identical
        if all(label == 0 for label in valid_labels[i]) or all(label == 1 for label in valid_labels[i]):
            continue
        results.append(metric_func(valid_labels[i], valid_preds[i]))

    return results


def evaluate(model: nn.Module,
             data: List[Tuple[str, List[float]]],
             metric_func: Callable,
             args: Namespace,
             scaler: StandardScaler = None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: Dataset.
    :param metric_func: Metric function which takes in a list of labels and a list of predictions.
    :param args: Arguments.
    :param scaler: A StandardScaler object fit on the training labels.
    :return: A list with the score for each task based on `metric_func`.
    """
    smiles, labels = zip(*data)

    preds = predict(
        model=model,
        smiles=smiles,
        args=args,
        scaler=scaler
    )

    results = evaluate_predictions(
        preds=preds,
        labels=labels,
        metric_func=metric_func
    )

    return results
