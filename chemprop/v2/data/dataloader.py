from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from chemprop.v2.data.datasets import Datum, MolGraphDatasetBase
from chemprop.v2.data.samplers import ClassBalanceSampler, SeededSampler
from chemprop.v2.featurizers.molgraph import BatchMolGraph

TrainingBatch = tuple[BatchMolGraph, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]


def collate_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, atom_descriptors, features, ys, weights, gt_targets, lt_targets = zip(*batch)

    return (
        BatchMolGraph(mgs),
        None if atom_descriptors[0] is None else torch.from_numpy(np.array(atom_descriptors, "f4")),
        None if features[0] is None else torch.from_numpy(np.array(features, "f4")),
        torch.from_numpy(np.array(ys, "f4")),
        torch.from_numpy(np.array(weights, "f4")).unsqueeze(1),
        None if lt_targets[0] is None else torch.from_numpy(np.array(lt_targets, "f4")),
        None if gt_targets[0] is None else torch.from_numpy(np.array(gt_targets, "f4")),
    )


class MolGraphDataLoader(DataLoader):
    """A `MolGraphDataLoader` is a PyTorch `DataLoader` for loading a `MolGraphDataset`

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
        dataset: MolGraphDatasetBase,
        batch_size: int = 50,
        num_workers: int = 0,
        class_balance: bool = False,
        seed: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.dset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        if self.class_balance:
            self.sampler = ClassBalanceSampler(self.dset.targets, seed, self.shuffle)
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

    @property
    def targets(self) -> np.ndarray:
        """the targets associated with each molecule"""
        if self.class_balance or self.shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )

        return np.array([self.dset.data[i].targets for i in self.sampler])

    @property
    def gt_targets(self) -> Optional[np.ndarray]:
        """whether each target is an inequality rather than a value target associated
        with each molecule"""
        if self.class_balance or self.shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )

        if self.dset.data[0].gt_targets is None:
            return None

        return np.array([self.dset.data[i].gt_targets for i in self.sampler])

    @property
    def lt_targets(self) -> Optional[np.ndarray]:
        """for whether each target is an inequality rather than a value target associated
        with each molecule"""
        if self.class_balance or self.shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )

        if self.dset.data[0].lt_targets is None:
            return None

        return np.array([self.dset.data[i].lt_targets for i in self.sampler])

    @property
    def iter_size(self) -> int:
        """the number of data points included in each full iteration of this dataloader."""
        return len(self.sampler)
