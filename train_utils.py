import math
from typing import Callable, Iterable, List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam

from mpn import mol2graph


def train(model: nn.Module,
          data: List[Tuple[str, List[float]]],
          batch_size: int,
          num_tasks: int,
          loss_func: Callable,
          optimizer: Adam,
          scaler: StandardScaler = None,
          three_d: bool = False):
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: Training data.
    :param batch_size: Batch size.
    :param num_tasks: Number of tasks.
    :param loss_func: Loss function.
    :param optimizer: Optimizer.
    :param scaler: A StandardScaler object fit on the training labels.
    :param three_d: Whether to include 3D information in atom and bond features.
    """
    model.train()

    loss_sum, num_iter = 0, 0
    for i in range(0, len(data), batch_size):
        # Prepare batch
        batch = data[i:i + batch_size]
        mol_batch, label_batch = zip(*batch)
        mol_batch = mol2graph(mol_batch, three_d=three_d)

        mask = torch.Tensor([[x is not None for x in lb] for lb in label_batch])
        labels = [[0 if x is None else x for x in lb] for lb in label_batch]
        if scaler is not None:
            labels = scaler.transform(labels)  # subtract mean, divide by std
        labels = torch.Tensor(labels)

        if next(model.parameters()).is_cuda:
            mask, labels = mask.cuda(), labels.cuda()

        # Run model
        model.zero_grad()
        preds = model(mol_batch)
        loss = loss_func(preds, labels) * mask
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


def predict_ensemble(models: Iterable[nn.Module],
                     smiles: List[str],
                     batch_size: int,
                     num_tasks: int,
                     scaler: StandardScaler = None,
                     three_d: bool = False) -> np.ndarray:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param models: An iterable over models.
    :param smiles: A list of smiles strings.
    :param batch_size: Batch size.
    :param num_tasks: Number of tasks.
    :param scaler: A StandardScaler object fit on the training labels.
    :param three_d: Whether to include 3D information in atom and bond features.
    :return: A 2D ndarray of predictions. The outer dimension is examples
    while the inner dimension is tasks.
    """
    with torch.no_grad():
        sum_preds = np.zeros((len(smiles), num_tasks))
        model_count = 0
        for model in models:
            model.eval()

            model_preds = []
            for i in range(0, len(smiles), batch_size):
                # Prepare batch
                mol_batch = smiles[i:i + batch_size]
                mol_batch = mol2graph(mol_batch, three_d=three_d)

                # Run model
                preds = model(mol_batch)
                preds = preds.data.cpu().numpy()
                if scaler is not None:
                    preds = scaler.inverse_transform(preds)

                model_preds.extend(preds.tolist())

            sum_preds += np.array(model_preds)
            model_count += 1

        avg_preds = sum_preds / model_count

        return avg_preds


def predict(model: nn.Module, *args, **kwargs):
    """Makes predictions on a dataset using a single model. See `predict_ensemble` for `args` and `kwargs`."""
    return predict_ensemble([model], *args, **kwargs)


def evaluate_ensemble(models: Iterable[nn.Module],
                      data: List[Tuple[str, List[float]]],
                      batch_size: int,
                      num_tasks: int,
                      metric_func: Callable,
                      scaler: StandardScaler = None,
                      three_d: bool = False) -> float:
    """
    Evaluates an ensemble of models on a dataset.

    :param models: A iterable over models.
    :param data: Dataset.
    :param batch_size: Batch size.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of labels and a list of predictions.
    :param scaler: A StandardScaler object fit on the training labels.
    :param three_d: Whether to include 3D information in atom and bond features.
    :return: Score based on `metric_func`.
    """
    with torch.no_grad():
        # all_preds and all_labels have shape (data_size, num_tasks)
        smiles, all_labels = zip(*data)
        all_preds = predict_ensemble(
            models=models,
            smiles=smiles,
            batch_size=batch_size,
            num_tasks=num_tasks,
            scaler=scaler,
            three_d=three_d
        )

        # Filter out empty labels
        # preds and labels have shape (num_tasks, data_size)
        preds = [[] for _ in range(num_tasks)]
        labels = [[] for _ in range(num_tasks)]
        for i in range(num_tasks):
            for j in range(len(data)):
                if all_labels[j][i] is not None:  # Skip those without labels
                    preds[i].append(all_preds[j][i])
                    labels[i].append(all_labels[j][i])

        # Compute metric
        results = []
        for i in range(num_tasks):
            # Skip if all labels are identical
            if all(label == 0 for label in labels[i]) or all(label == 1 for label in labels[i]):
                continue
            results.append(metric_func(labels[i], preds[i]))

        # Average
        result = sum(results) / len(results)

        return result


def evaluate(model: nn.Module, *args, **kwargs) -> float:
    """Evaluates a single model on a dataset. See `evaluate_ensemble` for `args` and `kwargs`."""
    return evaluate_ensemble([model], *args, **kwargs)
