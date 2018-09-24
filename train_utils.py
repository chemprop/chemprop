import math
from typing import Callable, List, Tuple

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam

from mpn import mol2graph


def train(data: List[Tuple[str, List[float]]],
          batch_size: int,
          num_tasks: int,
          model: nn.Module,
          loss_func: Callable,
          optimizer: Adam,
          scaler: StandardScaler = None,
          three_d: bool = False):
    """
    Trains a model for an epoch.

    :param data: Training data.
    :param batch_size: Batch size.
    :param num_tasks: Number of tasks.
    :param model: Model.
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


def evaluate(data: List[Tuple[str, List[float]]],
             batch_size: int,
             num_tasks: int,
             model: nn.Module,
             metric_func: Callable,
             scaler: StandardScaler = None,
             three_d: bool = False) -> float:
    """
    Evaluates a model on a dataset.

    :param data: Validation dataset.
    :param batch_size: Batch size.
    :param num_tasks: Number of tasks.
    :param model: Model.
    :param metric_func: Metric function which takes in a list of labels and a list of predictions.
    :param scaler: A StandardScaler object fit on the training labels.
    :param three_d: Whether to include 3D information in atom and bond features.
    :return: Score based on `metric_func`.
    """
    with torch.no_grad():
        model.eval()

        all_preds = [[] for _ in range(num_tasks)]
        all_labels = [[] for _ in range(num_tasks)]
        for i in range(0, len(data), batch_size):
            # Prepare batch
            batch = data[i:i + batch_size]
            mol_batch, label_batch = zip(*batch)
            mol_batch = mol2graph(mol_batch, three_d=three_d)
            mask = torch.Tensor([[x is not None for x in lb] for lb in label_batch])
            labels = [[0 if x is None else x for x in lb] for lb in label_batch]

            # Run model
            preds = model(mol_batch)
            preds = preds.data.cpu().numpy()
            if scaler is not None:
                preds = scaler.inverse_transform(preds)

            # Collect predictions and labels, skipping those without labels (i.e. masked out)
            for i in range(num_tasks):
                for j in range(len(batch)):
                    if mask[j][i] == 1:
                        all_preds[i].append(preds[j][i])
                        all_labels[i].append(labels[j][i])

        # Compute metric
        results = []
        for i in range(num_tasks):
            # Skip if all labels are identical
            if all(label == 0 for label in all_labels[i]) or all(label == 1 for label in all_labels[i]):
                continue
            results.append(metric_func(all_labels[i], all_preds[i]))

        # Average
        result = sum(results) / len(results)

        return result
