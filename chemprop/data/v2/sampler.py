from itertools import chain
from typing import Iterator, Optional

import numpy as np
from torch.utils.data import Sampler

from chemprop.data.v2.molecule import MolGraphDataset


class SeededSampler(Sampler):
    def __init__(self, dataset: MolGraphDataset, seed: int, shuffle: bool = False):
        super().__init__()

        if seed is None:
            raise ValueError("arg `seed` was `None`! A SeededSampler must be seeded!")

        self.idxs = np.arange(len(dataset))
        self.rg = np.random.default_rng(seed)
        self.shuffle = shuffle
        
    def __iter__(self) -> Iterator[int]:
        """an iterator over indices to sample."""
        if self.shuffle:
            self.rg.shuffle(self.idxs)

        return iter(self.idxs)

    def __len__(self) -> int:
        """the number of indices that will be sampled."""
        return len(self.idxs)

class ClassBalanceSampler(Sampler):
    """A `ClassBalanceSampler` samples data from a `MolGraphDataset` such that positive and 
    negative classes are equally sampled
    
    Parameters
    ----------
    dataset : MolGraphDataset
        the dataset from which to sample
    seed : int
        the random seed to use for shuffling (only used when `shuffle` is `True`)
    shuffle : bool, default=False
        whether to shuffle the data during sampling
    """
    def __init__(self, dataset: MolGraphDataset, seed: Optional[int] = None, shuffle: bool = False):
        super().__init__()

        self.shuffle = shuffle
        self.rg = np.random.default_rng(seed)

        idxs = np.arange(len(dataset))
        actives = np.array([targets.any() for _, targets in dataset])

        self.pos_idxs = idxs[actives]
        self.neg_idxs = idxs[~actives]

        self.length = 2 * min(len(self.pos_idxs), len(self.neg_idxs))

    def __iter__(self) -> Iterator[int]:
        """an iterator over indices to sample."""
        if self.shuffle:
            self.rg.shuffle(self.pos_idxs)
            self.rg.shuffle(self.neg_idxs)

        return chain(*zip(self.pos_idxs, self.neg_idxs))

    def __len__(self) -> int:
        """the number of indices that will be sampled."""
        return self.length
