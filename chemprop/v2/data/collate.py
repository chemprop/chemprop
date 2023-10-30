from dataclasses import dataclass, field, InitVar
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from chemprop.v2.data.datasets import Datum
from chemprop.v2.featurizers.molgraph import MolGraph


@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`s.

    It has all the attributes of a `MolGraph` with the addition of `a_scope` and `b_scope`. These
    define the respective atom- and bond-scope of each individual `MolGraph` within the
    `BatchMolGraph`. This class is intended for use with data loading, so it uses
    :obj:`~torch.Tensor`s to store data
    """

    mgs: InitVar[Sequence[MolGraph]]
    """A list of individual :class:`MolGraph`s to be batched together"""
    n_atoms: int = field(init=False)
    """the number of atoms in the batched graph"""
    n_bonds: int = field(init=False)
    """the number of bonds in the batched graph"""
    V: Tensor = field(init=False)
    """the atom feature matrix"""
    E: Tensor = field(init=False)
    """the bond feature matrix"""
    a2b: Tensor = field(init=False)
    """a mapping from atom index to indices of incoming bonds"""
    b2a: Tensor = field(init=False)
    """a mapping from bond index to index of the originating atom"""
    b2revb: Tensor = field(init=False)
    """A mapping from bond index to the index of the reverse bond."""
    a2a: Tensor | None = field(init=False)
    """a mapping from atom index to the indices of connected atoms"""
    b2b: Tensor | None = field(init=False, default=None)
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

            a2b.extend([self.n_bonds + b for b in mg.a2b[a]] for a in range(mg.n_atoms))
            for b in range(mg.n_bonds):
                b2a.append(self.n_atoms + mg.b2a[b])
                b2revb.append(self.n_bonds + mg.b2revb[b])

            self.a_scope.append(mg.n_atoms)
            self.b_scope.append(mg.n_bonds)

            self.n_atoms += mg.n_atoms
            self.n_bonds += mg.n_bonds

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()

        # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        pad_width = max(1, max(len(in_bonds) for in_bonds in a2b))
        self.a2b = torch.tensor(
            [a2b[a] + [0] * (pad_width - len(a2b[a])) for a in range(self.n_atoms)],
            dtype=torch.long,
        )
        self.b2a = torch.tensor(b2a, dtype=torch.long)
        self.b2revb = torch.tensor(b2revb, dtype=torch.long)
        self.a2a = self.b2a[self.a2b]

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`s in this batch"""
        return len(self.a_scope)

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.a2b = self.a2b.to(device)
        self.b2a = self.b2a.to(device)
        self.b2revb = self.b2revb.to(device)
        self.a2a = self.a2a.to(device)


TrainingBatch = tuple[BatchMolGraph, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]
MulticomponentTrainingBatch = tuple[
    list[BatchMolGraph], list[Tensor], Tensor, Tensor, Tensor | None, Tensor | None
]


def collate_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, V_ds, x_fs, ys, weights, lt_masks, gt_masks = zip(*batch)

    return (
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds, axis=0)).float(),
        None if x_fs[0] is None else torch.from_numpy(np.array(x_fs)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


def collate_multicomponent(batches: Iterable[Iterable[Datum]]) -> MulticomponentTrainingBatch:
    tbs = [collate_batch(batch) for batch in zip(*batches)]

    return MulticomponentTrainingBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_f,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )
