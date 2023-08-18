from dataclasses import InitVar, dataclass, field
from typing import NamedTuple, Sequence

import numpy as np
import torch


class MolGraph(NamedTuple):
    """A `MolGraph` represents the graph featurization of a molecule."""

    n_atoms: int
    """the number of atoms in the molecule"""
    n_bonds: int
    """the number of bonds in the molecule"""
    V: np.ndarray
    """an array of shape `V x d_v` containing the atom features of the molecule"""
    E: np.ndarray
    """an array of shape `E x d_e` containing the bond features of the molecule"""
    a2b: list[tuple[int]]
    """A list of length `V` that maps from an atom index to a list of incoming bond indices."""
    b2a: list[int]
    """A list of length `E` that maps from a bond index to the index of the atom the bond
    originates from."""
    b2revb: np.ndarray
    """A list of length `E` that maps from a bond index to the index of the reverse bond."""
    a2a: list[int] | None
    """a mapping from atom index to the indices of connected atoms"""
    b2b: np.ndarray | None
    """a mapping from bond index to the indices of connected bonds"""


@dataclass
class BatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`s.

    It has all the attributes of a `MolGraph` with the addition of `a_scope` and `b_scope`. These
    define the respective atom- and bond-scope of each individual `MolGraph` within the
    `BatchMolGraph`. This class is intended for use with data loading, so it uses
    :obj:`~torch.Tensor`s to store data

    NOTE: the `BatchMolGraph` does not currently possess a `b2b` attribute, so it is not a strict
    subclass of a `MolGraph`
    """

    mgs: InitVar[Sequence[MolGraph]]
    """A list of individual :class:`MolGraph`s to be batched together"""
    n_atoms: int = field(init=False)
    """the number of atoms in the batched graph"""
    n_bonds: int = field(init=False)
    """the number of bonds in the batched graph"""
    V: torch.Tensor = field(init=False)
    """the atom feature matrix"""
    E: torch.Tensor = field(init=False)
    """the bond feature matrix"""
    a2b: torch.Tensor = field(init=False)
    """a mapping from atom index to indices of incoming bonds"""
    b2a: torch.Tensor = field(init=False)
    """a mapping from bond index to index of the originating atom"""
    b2revb: torch.Tensor = field(init=False)
    """A mapping from bond index to the index of the reverse bond."""
    a2a: torch.Tensor | None = field(init=False)
    """a mapping from atom index to the indices of connected atoms"""
    b2b: torch.Tensor | None = field(init=False)
    """a mapping from bond index to the indices of connected bonds"""
    a_scope: list[int] = field(init=False)
    """the number of atoms for each molecule in the batch"""
    b_scope: list[int] = field(init=False)
    """the number of bonds for each molecule in the batch"""

    def __post_init__(self, mgs: Sequence[MolGraph]):
        self.n_atoms = 1
        self.n_bonds = 1
        self.a_scope = []
        self.b_scope = []

        # All start with zero padding so that padding indices with zeros returns a zero vector
        Vs = [np.zeros((1, mgs[0].V.shape[1]))]
        Es = [np.zeros((1, mgs[0].E.shape[1]))]
        a2b = [[]]
        b2a = [0]
        b2revb = [0]

        for mg in mgs:
            Vs.append(mg.V)
            Es.append(mg.E)

            a2b.extend([b + self.n_bonds for b in mg.a2b[a]] for a in range(mg.n_atoms))
            b2a.extend(self.n_atoms + mg.b2a[b] for b in range(mg.n_bonds))
            b2revb.extend(self.n_bonds + mg.b2revb[b] for b in range(mg.n_bonds))

            self.a_scope.append(mg.n_atoms)
            self.b_scope.append(mg.n_bonds)

            self.n_atoms += mg.n_atoms
            self.n_bonds += mg.n_bonds

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()

        # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))
        self.a2b = torch.tensor(
            [a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)],
            dtype=torch.long,
        )
        self.b2a = torch.tensor(b2a, dtype=torch.long)
        self.b2revb = torch.tensor(b2revb, dtype=torch.long)
        self.a2a = self.b2a[self.a2b]

    def __len__(self) -> int:
        """the number of individual `MolGraph`s in this batch"""
        return len(self.a_scope)

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.a2b = self.a2b.to(device)
        self.b2a = self.b2a.to(device)
        self.b2revb = self.b2revb.to(device)
        self.a2a = self.a2a.to(device)
