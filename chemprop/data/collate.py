from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor

from chemprop.data.datasets import Datum
from chemprop.data.molgraph import MolGraph, MolGraphPretrain


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


"""
To be conservative, the pretrain batch collate will be independent to non-pretrain version 
"""
@dataclass(repr=False, eq=False, slots=True)
class PreBatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraphPretrain`\s.

    It has all the attributes of a ``MolGraphPretrain`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data
    """

    mgs: Sequence[MolGraphPretrain]
    """A list of individual :class:`MolGraphPretrain`\s to be batched together"""
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
    batch_masked_atom_index: Tensor = field(init=False)
    """the index of the masked atom in batch version"""
    batch_masked_atom_label_list: Tensor = field(init=False)
    """the label of the masked atom in batch version"""

    __size: int = field(init=False)

    def __post_init__(self):
        """Only store the size of the input graphs, defer processing."""
        self.__size = len(self.mgs)

    def prepare_batch(self):
        """Processes the batch of `MolGraphPretrain`s to generate batched attributes."""
        Vs, Es, edge_indexes, rev_edge_indexes, batch_indexes = [], [], [], [], []
        batch_masked_atom_index, batch_masked_atom_label_list = [], []

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(self.mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))
            if mg.masked_atom_index is not None and mg.masked_atom_label_list is not None:
                batch_masked_atom_index.append(mg.masked_atom_index + num_nodes)
                batch_masked_atom_label_list.append(mg.masked_atom_label_list)

            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        if len(batch_masked_atom_index) > 0 and len(batch_masked_atom_label_list) > 0:
            self.batch_masked_atom_index = torch.from_numpy(np.concatenate(batch_masked_atom_index)).long()
            self.batch_masked_atom_label_list = torch.from_numpy(np.concatenate(batch_masked_atom_label_list)).long()

        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()

        return self

    def apply_mask(self, mask_atom_pre_percent: float):
        """Applies masking operations using `MolGraphPretrain` methods."""
        for mg in self.mgs:
            mg.masked_atom_pretraining(mask_atom_pre_percent)

        return self.prepare_batch()


    def apply_bond_deletion(self, mask_bond_pre_percent: float):
        """Applies bond deletion using `MolGraphPretrain` methods."""
        for mg in self.mgs:
            mg.bond_deletion_complete(mask_bond_pre_percent)

        return self.prepare_batch()


    def apply_subgraph_deletion(self, center: int, mask_subgraph_pre_percent: float):
        """Applies subgraph deletion using `MolGraphPretrain` methods."""
        for mg in self.mgs:
            mg.subgraph_deletion(center, mask_subgraph_pre_percent)

        return self.prepare_batch()


    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch_masked_atom_index = self.batch_masked_atom_index.to(device)
        self.batch_masked_atom_label_list = self.batch_masked_atom_label_list.to(device)
        self.batch = self.batch.to(device)

class PreTrainingBatch(NamedTuple):
    bmg: PreBatchMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_pre_batch(batch: Iterable[Datum]) -> PreTrainingBatch:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return PreTrainingBatch(
        PreBatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )