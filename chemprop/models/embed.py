from argparse import Namespace
from typing import Dict, List, Union

import torch
import torch.nn as nn
import numpy as np


class Embedding(nn.Module):
    """An embedding layers for encoding molecules."""

    def __init__(self,
                 args: Namespace,
                 smiles_map: Dict[str, int]):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(Embedding, self).__init__()
        self.cuda = args.cuda
        self.use_input_features = args.use_input_features

        self.map = smiles_map
        self.encoder = nn.Embedding(len(self.map), args.hidden_size)
        self.encoder.weight.data.uniform_(-0.001, 0.001)

    def forward(self,
                batch: List[str],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        batch = torch.as_tensor([self.map[x] for x in batch], dtype=torch.long)
        if self.cuda:
            batch = batch.cuda()
        mol_vecs = self.encoder(batch)

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            features_batch = features_batch.to(mol_vecs)

            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1,features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)

        return mol_vecs
