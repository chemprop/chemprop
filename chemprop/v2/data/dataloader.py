from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from chemprop.v2.data.datasets import Datum, _MolGraphDatasetMixin
from chemprop.v2.data.samplers import ClassBalanceSampler, SeededSampler
from chemprop.v2.featurizers.molgraph import BatchMolGraph

TrainingBatch = tuple[BatchMolGraph, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]
MulticomponentTrainingBatch = tuple[
    list[BatchMolGraph], list[Tensor], Tensor, Tensor, Tensor | None, Tensor | None
]


def collate_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, V_ds, x_fs, ys, weights, gt_masks, lt_masks = zip(*batch)

    return (
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds, axis=0)).float(),
        None if x_fs[0] is None else torch.from_numpy(np.array(x_fs)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


class MolGraphDataLoader(DataLoader):
    """A :class:`MolGraphDataLoader` is a :obj:`~torch.utils.data.DataLoader` for
    :class:`MolGraphDataset`s

    Parameters
    ----------
    dataset : MoleculeDataset
        The dataset containing the molecules to load.
    batch_size : int, default=50
        the batch size to load
    num_workers : int, default=0
        the number of workers used to build batches.
    class_balance : bool, default=False
        Whether to perform class balancing (i.e., use an equal number of positive and negative
        molecules). Class balance is only available for single task classification datasets. Set
        shuffle to True in order to get a random subset of the larger class.
    seed : int, default=None
        the random seed to use for shuffling (only used when `shuffle` is `True`)
    shuffle : bool, default=False
        whether to shuffle the data during sampling
    """

    def __init__(
        self,
        dataset: _MolGraphDatasetMixin,
        batch_size: int = 50,
        num_workers: int = 0,
        class_balance: bool = False,
        seed: int | None = None,
        shuffle: bool = True,
    ):
        self.dset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        if self.class_balance:
            self.sampler = ClassBalanceSampler(self.dset.Y, seed, self.shuffle)
        elif self.shuffle and seed is not None:
            self.sampler = SeededSampler(len(self.dset), seed)
        else:
            self.sampler = None

        super().__init__(
            self.dset,
            batch_size,
            self.sampler is None and self.shuffle,
            self.sampler,
            num_workers=num_workers,
            collate_fn=collate_batch,
        )

    # @property
    # def Y(self) -> np.ndarray:
    #     """the targets associated with each molecule"""
    #     if self.class_balance or self.shuffle:
    #         raise ValueError(
    #             "Cannot safely extract targets when class balance or shuffle are enabled."
    #         )

    #     return np.array([self.dset.data[i].y for i in self.sampler])

    # @property
    # def gt_mask(self) -> np.ndarray | None:
    #     """whether each target is an inequality rather than a value target associated
    #     with each molecule"""
    #     if self.class_balance or self.shuffle:
    #         raise ValueError(
    #             "Cannot safely extract targets when class balance or shuffle are enabled."
    #         )

    #     if self.dset.data[0].gt_mask is None:
    #         return None

    #     return np.array([self.dset.data[i].gt_mask for i in self.sampler])

    # @property
    # def lt_mask(self) -> np.ndarray | None:
    #     """for whether each target is an inequality rather than a value target associated
    #     with each molecule"""
    #     if self.class_balance or self.shuffle:
    #         raise ValueError(
    #             "Cannot safely extract targets when class balance or shuffle are enabled."
    #         )

    #     if self.dset.data[0].lt_mask is None:
    #         return None

    #     return np.array([self.dset.data[i].lt_mask for i in self.sampler])

    # @property
    # def iter_size(self) -> int:
    #     """the number of data points included in each full iteration of this dataloader."""
    #     return len(self.sampler)
