from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from rdkit import Chem

from chemprop.v2.featurizers.molgraph import MolGraph
from chemprop.v2.featurizers.atom import AtomFeaturizer
from chemprop.v2.featurizers.bond import BondFeaturizer


class MolGraphFeaturizerMixin(ABC):
    def __init__(
        self,
        atom_featurizer: Optional[AtomFeaturizer] = None,
        bond_featurizer: Optional[BondFeaturizer] = None,
        bond_messages: bool = True,
    ):
        self.atom_featurizer = atom_featurizer or AtomFeaturizer()
        self.bond_featurizer = bond_featurizer or BondFeaturizer()
        self.atom_fdim = len(self.atom_featurizer)
        self.bond_fdim = len(self.bond_featurizer)
        self.bond_messages = bond_messages

        
    def __call__(self, *args, **kwargs):
        return self.featurize(*args, **kwargs)

    @abstractmethod
    def featurize(
        self,
        mol_or_reaction: Union[Chem.Mol, tuple[Chem.Mol, Chem.Mol]],
        atom_features_extra: Optional[np.ndarray] = None,
        bond_features_extra: Optional[np.ndarray] = None,
    ) -> MolGraph:
        pass


class MultiHotFeaturizer(ABC):
    """A `MultiHotFeaturizer` calculates feature vectors of arbitrary objects by concatenating
    multiple one-hot feature vectors"""

    def __call__(self, x) -> np.ndarray:
        return self.featurize(x)

    @abstractmethod
    def __len__(self) -> int:
        """the length of a feature vector from this featurizer"""

    @property
    @abstractmethod
    def subfeatures(self) -> Mapping[str, slice]:
        """a map from subfeature name to the slice in the output feature vectors"""

    @abstractmethod
    def featurize(self, x) -> np.ndarray:
        """calculate the feature vector of x"""

    @property
    def num_subfeatures(self) -> int:
        return len(self.subfeatures)

    @staticmethod
    def one_hot_index(x, xs: Sequence) -> tuple[int, int]:
        """return the index of a one hot encoding of `x` given choices `xs` and the length of the
        uncompressed encoding"""
        n = len(xs)
        return xs.index(x) if x in xs else n, n + 1