from abc import abstractmethod
from collections.abc import Sequence
from typing import Generic, Iterable, TypeVar

from rdkit import Chem
import numpy as np

from chemprop.featurizers.molgraph.base import MolGraphFeaturizer, T
from chemprop.featurizers.molgraph.molgraph import MolGraph

T = TypeVar("T", Chem.Mol, tuple[Chem.Mol, Chem.Mol])


class MolGraphCacheFacade(Sequence[MolGraph], Generic[T]):
    @abstractmethod
    def __init__(
        self,
        inputs: Iterable[T],
        V_fs: Iterable[np.ndarray],
        E_fs: Iterable[np.ndarray],
        featurizer: MolGraphFeaturizer[T],
    ):
        pass


class MolGraphCache(MolGraphCacheFacade):
    def __init__(
        self,
        inputs: Iterable[T],
        V_fs: Iterable[np.ndarray | None],
        E_fs: Iterable[np.ndarray | None],
        featurizer: MolGraphFeaturizer[T],
    ):
        self._mgs = [featurizer(input, V_f, E_f) for input, V_f, E_f in zip(inputs, V_fs, E_fs)]

    def __len__(self) -> int:
        return len(self._mgs)

    def __getitem__(self, index: int) -> MolGraph:
        return self._mgs[index]


class MolGraphCacheOnTheFly(MolGraphCacheFacade):
    def __init__(
        self,
        inputs: Iterable[T],
        V_fs: Iterable[np.ndarray | None],
        E_fs: Iterable[np.ndarray | None],
        featurizer: MolGraphFeaturizer[T],
    ):
        self._inputs = list(inputs)
        self._V_fs = list(V_fs)
        self._E_fs = list(E_fs)
        self._featurizer = featurizer

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, index: int) -> MolGraph:
        return self._featurizer(self._inputs[index], self._V_fs[index], self._E_fs[index])
