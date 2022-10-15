from __future__ import annotations
from ctypes import Union

from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Optional

import numpy as np
import torch
from torch import Tensor


class MolGraph(NamedTuple):
    """A `MolGraph` represents the graph structure and featurization of a single molecule.

    Attributes
    ----------
    n_atoms : int
        the number of atoms in the molecule
    n_bonds : int
        the number of bonds in the molecule
    X_v : np.ndarray
        an array of shape `V x d_v` containing the atom features of the molecule
    X_e : np.ndarray
        an array of shape `E x d_e` containing the bond features of the molecule
    a2b : list[tuple[int]]
        A list of length `V` that maps from an atom index to a list of incoming bond indices.
    b2a : np.ndarray
        A list of length `E` that maps from a bond index to the index of the atom the bond
        originates from.
    b2revb : np.ndarray
        A list of length `E` that maps from a bond index to the index of the reverse bond.
    a2a : Optional[list[int]]
    b2b: Optional[np.ndarray]
    """

    n_atoms: int
    n_bonds: int
    X_v: np.ndarray
    X_e: np.ndarray
    a2b: list[tuple[int]]
    b2a: list[int]
    b2revb: np.ndarray
    a2a: Optional[list[int]]
    b2b: Optional[np.ndarray]


@dataclass
class BatchMolGraph:
    """A `BatchMolGraph` is a singular `MolGraph` composed of a batch of individual `MolGraphs`.

    It has all the attributes of a `MolGraph` with the addition of `a_scope` and `b_scope`. These
    define the atom and bond-scope of each original `MolGraph` in the batched `MolGraph`,
    respectively. This class is intended for use with data loading, so it uses `Tensors`s to
    store data

    NOTE: the `BatchMolGraph` does not currently possess a `b2b` attribute, so it is not a strict
    subclass of a `MolGraph`

    Attributes
    ----------
    n_atoms : int
    n_bonds : int
    X_v : Tensor
    X_e : Tensor
    a2b : Tensor
    b2a : Tensor
    b2revb : Tensor
    a2a : Tensor
    a_scope : list[tuple[int]]
        a list of tuples containing `(start_index, n_atoms)` for each molecule in the batch
    b_scope : list[tuple[int]]
        a list of tuples containing `(start_index, n_bonds)` for each molecule in the batch
    """

    mgs: InitVar[Iterable[MolGraph]]
    n_atoms: int = field(init=False)
    n_bonds: int = field(init=False)
    X_v: Tensor = field(init=False)
    X_e: Tensor = field(init=False)
    a2b: Tensor = field(init=False)
    b2a: Tensor = field(init=False)
    b2revb: Tensor = field(init=False)
    a2a: Tensor = field(init=False)
    # b2b: Optional[np.ndarray] = field(init=False)
    a_scope: list[tuple[int, int]] = field(init=False)
    b_scope: list[tuple[int, int]] = field(init=False)

    def __post_init__(self, mgs: Iterable[MolGraph]):
        self.n_atoms = 1
        self.n_bonds = 1
        self.a_scope = []
        self.b_scope = []

        # All start with zero padding so that indexing with zero padding returns zeros
        X_vs = [np.zeros((1, mgs[0].X_v.shape[1]))]
        X_es = [np.zeros((1, mgs[0].X_e.shape[1]))]
        a2b = [[]]
        b2a = [0]
        b2revb = [0]

        for mg in mgs:
            X_vs.append(mg.X_v)
            X_es.append(mg.X_e)

            a2b.extend([b + self.n_bonds for b in mg.a2b[a]] for a in range(mg.n_atoms))
            b2a.extend(self.n_atoms + mg.b2a[b] for b in range(mg.n_bonds))
            b2revb.extend(self.n_bonds + mg.b2revb[b] for b in range(mg.n_bonds))

            self.a_scope.append((self.n_atoms, mg.n_atoms))
            self.b_scope.append((self.n_bonds, mg.n_bonds))

            self.n_atoms += mg.n_atoms
            self.n_bonds += mg.n_bonds

        self.X_v = torch.from_numpy(np.concatenate(X_vs)).float()
        self.X_e = torch.from_numpy(np.concatenate(X_es)).float()

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
        return len(self.a_scope)

    def to(self, device: Union[str, torch.device]):
        self.X_v = self.X_v.to(device)
        self.X_e = self.X_e.to(device)
        self.a2b = self.a2b.to(device)
        self.b2a = self.b2a.to(device)
        self.b2revb = self.b2revb.to(device)
        self.a2a = self.a2a.to(device)
