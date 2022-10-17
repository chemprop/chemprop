from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from rdkit import Chem

from chemprop.v2.featurizers.molgraph import MolGraph
from chemprop.v2.featurizers.atom import AtomFeaturizer, AtomFeaturizerBase
from chemprop.v2.featurizers.bond import BondFeaturizer, BondFeaturizerBase


class MolGraphFeaturizerMixin(ABC):
    def __init__(
        self,
        atom_featurizer: Optional[AtomFeaturizerBase] = None,
        bond_featurizer: Optional[BondFeaturizerBase] = None,
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
