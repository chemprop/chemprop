from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np


class MultiHotFeaturizer(ABC):
    def __call__(self, x) -> np.ndarray:
        return self.featurize(x)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def offsets(self) -> Iterable[int]:
        pass

    @abstractmethod
    def featurize(self, x) -> np.ndarray:
        pass

    @property
    def num_classes(self) -> int:
        return len(self.offsets)

    @staticmethod
    def safe_index(x, xs: Sequence):
        """return the index of `x` in `xs` if it exists, otherwise return -1"""
        return xs.index(x) if x in xs else -1
