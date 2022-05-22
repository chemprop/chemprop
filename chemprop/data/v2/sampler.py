from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

from chemprop.data.v2.data import MolGraphDataset

class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a
    :class:`MoleculeDataLoader`."""

    def __init__(
        self,
        dataset: MolGraphDataset,
        class_balance: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super().__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = np.random.default_rng(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array(
                [any(target == 1 for target in datapoint.targets) for datapoint in dataset]
            )

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [
                index
                for pair in zip(self.positive_indices, self.negative_indices)
                for index in pair
            ]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """the number of indices that will be sampled."""
        return self.length