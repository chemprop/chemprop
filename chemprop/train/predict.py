from argparse import Namespace
import logging
from typing import List

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.multiprocessing import Process, Queue
import numpy as np
from tqdm import trange

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.features import mol2graph
from chemprop.models import build_model
from chemprop.utils import get_loss_func


def predict(model: nn.Module,
            data: MoleculeDataset,
            args: Namespace,
            scaler: StandardScaler = None,
            logger: logging.Logger = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param args: Arguments.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []

    num_iters, iter_step = len(data), args.batch_size

    for i in trange(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        with torch.no_grad():
            batch_preds = model(batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)
        
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
