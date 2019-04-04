from typing import List, Union

import numpy as np
import torch
import torch.nn as nn

from chemprop.features import BatchMolGraph


class FeaturesOnly(nn.Module):
    """A model which ignores the molecules and returns only the pre-computed features."""

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        Ignores the molecules and returns just the pre-computed features for each molecule.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, features_size) containing the features of each molecule.
        """
        return torch.FloatTensor(features_batch)
