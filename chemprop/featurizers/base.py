from abc import abstractmethod
from collections.abc import Sized
from typing import Generic, TypeVar

import numpy as np

from chemprop.data.molgraph import MolGraph, WeightedMolGraph

S = TypeVar("S")
T = TypeVar("T")


class Featurizer(Generic[S, T]):
    """An :class:`Featurizer` featurizes inputs type ``S`` into outputs of
    type ``T``."""

    @abstractmethod
    def __call__(self, input: S, *args, **kwargs) -> T:
        """featurize an input"""


class VectorFeaturizer(Featurizer[S, np.ndarray], Sized):
    ...


class GraphFeaturizer(Featurizer[S, MolGraph | WeightedMolGraph]):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        ...
