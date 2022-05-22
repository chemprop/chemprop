from __future__ import annotations
import threading
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from chemprop.data.v2.data import MolGraphDataset
from chemprop.data.v2.sampler import MoleculeSampler
from chemprop.featurizers.molgraph import MolGraph


def collate_graphs(mgs: Sequence[MolGraph]) -> tuple:
    n_atoms = 1
    n_bonds = 1
    a_scope = []
    b_scope = []

    # All start with zero padding so that indexing with zero padding returns zeros
    X_vs = [np.zeros(mgs[0].X_v.shape[0])]
    X_es = [np.zeros(mgs[0].X_e.shape[0])]
    a2b = [[]]
    b2a = [0]
    b2revb = [0]

    for mg in mgs:
        X_vs.append(mg.X_v)
        X_es.append(mg.X_v)

        for a in range(mg.n_atoms):
            a2b.append([b + n_bonds for b in mg.a2b[a]])

        for b in range(mg.n_bonds):
            b2a.append(n_atoms + mg.b2a[b])
            b2revb.append(n_bonds + mg.b2revb[b])

        a_scope.append((n_atoms, mg.n_atoms))
        b_scope.append((n_bonds, mg.n_bonds))
        n_atoms += mg.n_atoms
        n_bonds += mg.n_bonds


    X_v = torch.cat(X_vs)
    X_e = torch.cat(X_vs)

    # max with 1 to fix a crash in rare case of all single-heavy-atom mols
    max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))
    a2b = torch.tensor(
        [a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(n_atoms)], dtype=torch.long
    )

    b2a = torch.tensor(b2a, dtype=torch.long)
    b2revb = torch.tensor(b2revb, dtype=torch.long)
    b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
    a2a = None  # only needed if using atom messages

    return X_v, X_e, a2b, b2a, b2revb, a_scope, b_scope
 

class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(
        self,
        dataset: MolGraphDataset,
        batch_size: int = 50,
        num_workers: int = 0,
        class_balance: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of
            positive and negative molecules). Class balance is only available for single task
            classification datasets. Set shuffle to True in order to get a random subset of the
            larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = "forkserver"  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            self._dataset,
            self._class_balance,
            self._shuffle,
            self._seed,
        )

        super().__init__(
            self._dataset,
            self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=collate_graphs,
            multiprocessing_context=self._context,
            timeout=self._timeout,
        )

    @property
    def targets(self) -> list[list[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def gt_targets(self) -> list[list[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )

        if not hasattr(self._dataset[0], "gt_targets"):
            return None

        return [self._dataset[index].gt_targets for index in self._sampler]

    @property
    def lt_targets(self) -> list[list[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )

        if not hasattr(self._dataset[0], "lt_targets"):
            return None

        return [self._dataset[index].lt_targets for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)
