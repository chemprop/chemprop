from enum import auto
from typing import Sequence

import numpy as np

from chemprop.v2.utils import AutoName


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


class ReactionMode(AutoName):
    """The manner in which a reaction should be featurized into a `MolGraph`"""

    REAC_PROD = auto()
    """concatenate the reactant features with the product features."""
    REAC_PROD_BALANCE = auto()
    """concatenate the reactant features with the products feature and balances imbalanced
    reactions"""
    REAC_DIFF = auto()
    """concatenates the reactant features with the difference in features between reactants and
    products"""
    REAC_DIFF_BALANCE = auto()
    """concatenates the reactant features with the difference in features between reactants and
    product and balances imbalanced reactions"""
    PROD_DIFF = auto()
    """concatenates the product features with the difference in features between reactants and
    products"""
    PROD_DIFF_BALANCE = auto()
    """concatenates the product features with the difference in features between reactants and
    products and balances imbalanced reactions"""
