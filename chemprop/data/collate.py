from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor

from chemprop.data.datasets import Datum, MolAtomBondDatum
from chemprop.data.molgraph import MolGraph


@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`\s.

    It has all the attributes of a ``MolGraph`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data
    """

    mgs: InitVar[Sequence[MolGraph]]
    """A list of individual :class:`MolGraph`\s to be batched together"""
    V: Tensor = field(init=False)
    """the atom feature matrix"""
    E: Tensor = field(init=False)
    """the bond feature matrix"""
    edge_index: Tensor = field(init=False)
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor = field(init=False)
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
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

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))

            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)


class TrainingBatch(NamedTuple):
    bmg: BatchMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


class MolAtomBondTrainingBatch(NamedTuple):
    bmg: BatchMolGraph
    V_d: Tensor | None
    E_d: Tensor | None
    X_d: Tensor | None
    Ys: list[Tensor | None]
    w: Tensor
    lt_masks: list[Tensor | None]
    gt_masks: list[Tensor | None]


def collate_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return TrainingBatch(
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


def collate_mol_atom_bond_batch(batch: Iterable[MolAtomBondDatum]) -> MolAtomBondTrainingBatch:
    mgs, V_ds, E_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    np_ys, np_lts, np_gts, weights_tensors = [], [], [], []
    for dataset in zip(*ys):
        if dataset[0] is None:
            np_ys.append(None)
            weights_tensors.append(None)
            continue
        else:
            np_ys.append(np.vstack(dataset))
        if np_ys[-1].shape[0] == len(dataset):
            weights_tensors.append(torch.tensor(weights, dtype=torch.float).unsqueeze(1))
        else:
            num_atoms = torch.tensor([y.shape[0] for y in dataset])
            tensor = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
            weights_tensors.append(torch.repeat_interleave(tensor, repeats=num_atoms))

    for dataset in zip(*lt_masks):
        np_lts.append(np.vstack(dataset))
    for dataset in zip(*gt_masks):
        np_gts.append(np.vstack(dataset))

    return MolAtomBondTrainingBatch(
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if E_ds[0] is None else torch.from_numpy(np.concatenate(E_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        [torch.from_numpy(np_y).float() for np_y in np_ys],
        weights_tensors,
        [None if lt_masks[0][i] is None else torch.from_numpy(np_lts[i]) for i in range(3)],
        [None if gt_masks[0][i] is None else torch.from_numpy(np_gts[i]) for i in range(3)],
    )


class MulticomponentTrainingBatch(NamedTuple):
    bmgs: list[BatchMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_multicomponent(batches: Iterable[Iterable[Datum]]) -> MulticomponentTrainingBatch:
    tbs = [collate_batch(batch) for batch in zip(*batches)]

    return MulticomponentTrainingBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )
