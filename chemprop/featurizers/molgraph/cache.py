from abc import abstractmethod
from collections.abc import Sequence
from typing import Generic, Iterable

import numpy as np

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import Featurizer, S
from chemprop.utils import parallel_execute


class MolGraphCacheFacade(Sequence[MolGraph], Generic[S]):
    """
    A :class:`MolGraphCacheFacade` provided an interface for caching
    :class:`~chemprop.data.molgraph.MolGraph`\s.

    .. note::
        This class only provides a facade for a cached dataset, but it *does not guarantee*
        whether the underlying data is truly cached.


    Parameters
    ----------
    inputs : Iterable[S]
        The inputs to be featurized.
    V_fs : Iterable[np.ndarray]
        The node features for each input.
    E_fs : Iterable[np.ndarray]
        The edge features for each input.
    featurizer : Featurizer[S, MolGraph]
        The featurizer with which to generate the
        :class:`~chemprop.data.molgraph.MolGraph`\s.
    """

    @abstractmethod
    def __init__(
        self,
        inputs: Iterable[S],
        V_fs: Iterable[np.ndarray],
        E_fs: Iterable[np.ndarray],
        featurizer: Featurizer[S, MolGraph],
    ):
        pass


class MolGraphCache(MolGraphCacheFacade):
    """
    A :class:`MolGraphCache` precomputes the corresponding
    :class:`~chemprop.data.molgraph.MolGraph`\s and caches them in memory.
    """

    def __init__(
        self,
        inputs: Iterable[S],
        V_fs: Iterable[np.ndarray | None],
        E_fs: Iterable[np.ndarray | None],
        featurizer: Featurizer[S, MolGraph],
        n_workers: int = 0,
    ):
        self._mgs = parallel_execute(featurizer, zip(inputs, V_fs, E_fs), n_workers=n_workers)

    def __len__(self) -> int:
        return len(self._mgs)

    def __getitem__(self, index: int) -> MolGraph:
        return self._mgs[index]


class MolGraphCacheOnTheFly(MolGraphCacheFacade):
    """
    A :class:`MolGraphCacheOnTheFly` computes the corresponding
    :class:`~chemprop.data.molgraph.MolGraph`\s as they are requested.
    """

    def __init__(
        self,
        inputs: Iterable[S],
        V_fs: Iterable[np.ndarray | None],
        E_fs: Iterable[np.ndarray | None],
        featurizer: Featurizer[S, MolGraph],
    ):
        self._inputs = list(inputs)
        self._V_fs = list(V_fs)
        self._E_fs = list(E_fs)
        self._featurizer = featurizer

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, index: int) -> MolGraph:
        return self._featurizer(self._inputs[index], self._V_fs[index], self._E_fs[index])
