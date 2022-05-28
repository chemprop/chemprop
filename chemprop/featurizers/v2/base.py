from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from rdkit import Chem

from chemprop.featurizers.v2.molgraph import MolGraph


class MolGraphFeaturizer(ABC):
    def __call__(self, *args, **kwargs) -> MolGraph:
        return self.featurize(*args, **kwargs)

    @abstractmethod
    def featurize(
        self,
        mol_or_reaction: Union[Chem.Mol, list[Chem.Mol]],
        atom_features_extra: Optional[np.ndarray] = None,
        bond_features_extra: Optional[np.ndarray] = None,
    ) -> MolGraph:
        pass
