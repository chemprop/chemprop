from dataclasses import InitVar, dataclass, field
from typing import NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor

class MolGraph(NamedTuple):
    """A :class:`MolGraph` represents the graph featurization of a molecule."""

    n_atoms: int
    """the number of atoms in the molecule"""
    n_bonds: int
    """the number of bonds in the molecule"""
    V: np.ndarray
    """an array of shape ``V x d_v`` containing the atom features of the molecule"""
    E: np.ndarray
    """an array of shape ``2 * E x d_e`` containing the bond features of the molecule"""
    edge_index: np.ndarray
    """an array of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: np.ndarray
    """A vector of length ``E`` that maps from an edge index to the index of the source of the reverse edge in the ``edge_index`` attribute."""


@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`s.

    It has all the attributes of a ``MolGraph`` with the addition of `a_scope` and `b_scope`. These
    define the respective atom- and bond-scope of each individual `MolGraph` within the
    `BatchMolGraph`. This class is intended for use with data loading, so it uses
    :obj:`~torch.Tensor`s to store data
    """

    mgs: InitVar[Sequence[MolGraph]]
    """A list of individual :class:`MolGraph`s to be batched together"""
    V: Tensor = field(init=False)
    """the atom feature matrix"""
    E: Tensor = field(init=False)
    """the bond feature matrix"""
    edge_index: Tensor = field(init=False)
    """an array of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor = field(init=False)
    """A vector of length ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    atom_edge_index: Tensor | None = field(init=False)
    """an array of shape ``2 x V`` containing the edges of the atom-based graph in COO format"""
    batch: Tensor = field(init=False)
    """the index of the parent :class:`MolGraph` in the batched graph"""

    __size: int = field(init=False)

    def __post_init__(self, mgs: Sequence[MolGraph]):
        self.__size = len(mgs)
        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []

        offset = 0
        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + offset)
            rev_edge_indexes.append(mg.rev_edge_index + offset)
            batch_indexes.append([i] * len(mg.V))

            offset += mg.edge_index.max(initial=0) + 1

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes))
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes))
        self.batch = torch.from_numpy(np.concatenate(batch_indexes))
    
    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)
