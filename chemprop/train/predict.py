from typing import List
import csv
import os

import torch
import torch.nn as nn
from tqdm import trange

from chemprop.data import MolPairDataset, StandardScaler


def predict(model: nn.Module,
            data: MolPairDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MolPairDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []

    num_iters, iter_step = len(data), batch_size

    for i in trange(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MolPairDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        with torch.no_grad():
            batch_preds = model(batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds

def save_predictions(save_dir: str,
                     train_data: MolPairDataset,
                     val_data: MolPairDataset,
                     test_data: MolPairDataset,
                     train_preds: List[List[float]],
                     val_preds: List[List[float]],
                     test_preds: List[List[float]]) -> None:
    """
    Saves predictions to csv file for entire model.
    """
    with open(os.path.join(save_dir, 'preds.csv'), 'w') as f:
        writer = csv.writer(f)
        header = ['drugSMILE', 'cmpdSMILE', 'split', 'truth', 'pred']
        writer.writerow(header)

        splits = ['train', 'val', 'test']
        dataSplits = [train_data, val_data, test_data]
        predSplits = [train_preds, val_preds, test_preds]
        for k, split in enumerate(splits):
            smiles = dataSplits[k].smiles()
            targets = dataSplits[k].targets()
            preds = predSplits[k]
            for i in range(len(smiles)):
                row = [smiles[i][0], smiles[i][1], split, targets[i][0], preds[i][0]]
                writer.writerow(row)
