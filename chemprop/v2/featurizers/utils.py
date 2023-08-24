from typing import Sequence

import numpy as np



class MultiHotFeaturizerMixin:
    """A `MultiHotFeaturizer` calculates feature vectors of arbitrary objects by concatenating
    multiple one-hot feature vectors"""

    def __call__(self, x) -> np.ndarray:
        return self.featurize(x)

    @property
    def num_subfeatures(self) -> int:
        return len(self.subfeatures)

    @staticmethod
    def one_hot_index(x, xs: Sequence) -> tuple[int, int]:
        """the index of `x` in `xs`, if it exists. Otherwise, return `len(xs) + 1`."""
        n = len(xs)

        return xs.index(x) if x in xs else n, n + 1
