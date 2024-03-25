from abc import abstractmethod
from collections.abc import Callable
from typing import Generic, Sized, TypeVar

import numpy as np

from chemprop.data.molgraph import MolGraph

S = TypeVar("S")
T = TypeVar("T")


class Featurizer(Generic[S, T]):
    """An :class:`Featurizer` featurizes inputs type ``S`` into outputs of
    type ``T``."""

    # @abstractmethod
    # def __len__(self) -> int:
    #     """the length of an atomic feature vector"""

    @abstractmethod
    def __call__(self, input: S, *args, **kwargs) -> T:
        """featurize an input"""


class VectorFeaturizer(Featurizer[S, np.ndarray], Sized):
    ...


class GraphFeaturizer(Featurizer[S, MolGraph]):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]: ...