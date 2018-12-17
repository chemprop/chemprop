from argparse import Namespace
from typing import List

import torch
import torch.nn as nn
import numpy as np

from chemprop.data import MoleculeDataset, StandardScaler


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
