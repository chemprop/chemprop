from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

class MultiHotFeaturizer(ABC):
    def __call__(self, x) -> np.ndarray:
        return self.featurize(x)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def featurize(self, x) -> np.ndarray:
        pass

    @staticmethod
    def safe_index(x, xs: Sequence):
        """return both the index of `x` in `xs` (if it exists, else -1) and the total length of `xs`"""
        return xs.index(x) if x in xs else len(xs), len(xs)
