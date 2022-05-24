from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np


class MultiHotFeaturizer(ABC):
    """A MultiHotFeaturizer calculates feature vectors of arbitrary objects by concatenation of
    multiple feature vectors
    
    NOTE: classes that implement the MultiHotFeaturizer should call `super().__init__() at the *end* of their respective `__init__()`
    """
    def __init__(self):
        self._validate_subfeatures()

    def __call__(self, x) -> np.ndarray:
        return self.featurize(x)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def subfeatures(self) -> Mapping[str, int]:
        """a map from subfeature alias to the start index in the calculated feature vector.
        
        NOTE: the ordering of the subfeatures in this dictionary is in increasing order such that
        the indices of each subfeature are can be calculated like so:
        >>> offsets = list(self.subfeatures.values())
        >>> slices = [slice(i, j) for i, j in zip(offsets, offsets[1:])]
        """

    @abstractmethod
    def featurize(self, x) -> np.ndarray:
        pass

    @property
    def num_subfeatures(self) -> int:
        return len(self.subfeatures)

    def _validate_subfeatures(self):
        offsets = list(self.subfeatures.values())
        if not all(i < j for i, j in zip(offsets, offsets[1:])):
            raise ValueError("Improperly formatted `subfeatures` mapping!")

    @staticmethod
    def safe_index(x, xs: Sequence):
        """return the index of `x` in `xs` if it exists, otherwise return -1"""
        return xs.index(x) if x in xs else -1
