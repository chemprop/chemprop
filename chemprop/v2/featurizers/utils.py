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
        """return the index of a one hot encoding of `x` given choices `xs` and the length of the
        uncompressed encoding"""
        n = len(xs)

        return xs.index(x) if x in xs else n, n + 1


class ReactionMode(AutoName):
    """The manner in which a reaction should be featurized into a `MolGraph`

    REAC_PROD
        concatenate the reactant features with the product features.
    REAC_PROD_BALANCE
        concatenate the reactant features with the products feature and balances imbalanced
        reactions.
    REAC_DIFF
        concatenates the reactant features with the difference in features between reactants and
        products
    REAC_DIFF_BALANCE
        concatenates the reactant features with the difference in features between reactants and
        products and balances imbalanced reactions
    PROD_DIFF
        concatenates the product features with the difference in features between reactants and
        products
    PROD_DIFF_BALANCE
        concatenates the product features with the difference in features between reactants and
        products and balances imbalanced reactions
    """

    REAC_PROD = auto()
    REAC_PROD_BALANCE = auto()
    REAC_DIFF = auto()
    REAC_DIFF_BALANCE = auto()
    PROD_DIFF = auto()
    PROD_DIFF_BALANCE = auto()
