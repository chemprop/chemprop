from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np


class MultiHotFeaturizer(ABC):
    """A `MultiHotFeaturizer` calculates feature vectors of arbitrary objects by concatenating
    multiple one-hot feature vectors"""

    def __call__(self, x) -> np.ndarray:
        return self.featurize(x)

    @abstractmethod
    def __len__(self) -> int:
        """the length of a feature vector from this featurizer"""

    @property
    @abstractmethod
    def subfeatures(self) -> Mapping[str, slice]:
        """a map from subfeature name to the slice in the output feature vectors"""

    @abstractmethod
    def featurize(self, x) -> np.ndarray:
        """calculate the feature vector of x"""

    @property
    def num_subfeatures(self) -> int:
        return len(self.subfeatures)

    @staticmethod
    def one_hot_index(x, xs: Sequence) -> tuple[int, int]:
        """return the index of a one hot encoding of `x` given choices `xs` and the length of the
        uncompressed encoding"""
        n = len(xs)
        return xs.index(x) if x in xs else n, n + 1
