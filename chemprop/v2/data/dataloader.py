from typing import Iterable, NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from chemprop.v2.data.datasets import Datum, MoleculeDataset, ReactionDataset, MulticomponentDataset
from chemprop.v2.data.samplers import ClassBalanceSampler, SeededSampler
from chemprop.v2.featurizers.molgraph import BatchMolGraph


class TrainingBatch(NamedTuple):
    bmg: BatchMolGraph
    V_d: Tensor | None
    X_f: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


class MulticomponentTrainingBatch(NamedTuple):
    bmgs: list[BatchMolGraph]
    V_ds: list[Tensor | None]
    X_f: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, V_ds, x_fs, ys, weights, lt_masks, gt_masks = zip(*batch)

    return TrainingBatch(
        BatchMolGraph(mgs),
        None if np.equal(V_ds, None).all() else torch.from_numpy(np.vstack(V_ds)).float(),
        None if np.equal(x_fs, None).all() else torch.from_numpy(np.array(x_fs)).float(),
        None if np.isnan(ys).all() else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights).unsqueeze(1),
        None if np.equal(lt_masks, None).all() else torch.from_numpy(np.array(lt_masks)),
        None if np.equal(gt_masks, None).all() else torch.from_numpy(np.array(gt_masks)),
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
        dataset: MoleculeDataset | ReactionDataset | MulticomponentDataset,
        batch_size: int = 50,
        num_workers: int = 0,
        class_balance: bool = False,
        seed: int | None = None,
        shuffle: bool = True,
        **kwargs,
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

        if isinstance(dataset, MulticomponentDataset):
            collate_fn = collate_multicomponent
        else:
            collate_fn = collate_batch

        super().__init__(
            self.dset,
            batch_size,
            self.sampler is None and self.shuffle,
            self.sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
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
