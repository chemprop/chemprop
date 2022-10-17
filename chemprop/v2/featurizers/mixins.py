from abc import abstractmethod
from typing import Optional, Union

import numpy as np
from rdkit import Chem

from chemprop.v2.featurizers.molgraph import MolGraph
from chemprop.v2.featurizers.multihot.atom import AtomFeaturizer
from chemprop.v2.featurizers.multihot.bond import BondFeaturizer


class MolGraphFeaturizerMixin:
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