from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
from rdkit.Chem.rdchem import Bond, BondType

from chemprop.v2.featurizers.multihot import MultiHotFeaturizer


class BondFeaturizerBase(ABC):
    """An `AtomFeaturizerBase` calculates feature vectors of RDKit atoms."""

    @abstractmethod
    def __call__(self, b: Bond) -> np.ndarray:
        """featurize the atom `b`"""


class BondFeaturizer(MultiHotFeaturizer):
    """A `BondFeaturizer` generates multihot featurizations of RDKit bonds

    Parameters
    ----------
    bond_types : Optional[Sequence[BondType]], default=[SINGLE, DOUBLE, TRIPLE, AROMATIC]
        the known bond types
    stereos : Optional[Sequence[int]], default=[0, 1, 2, 3, 4, 5]
        the known bond stereochemistries. See [1]_ for more details
    
    References
    ----------
    .. [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondStereo.values
    """

    def __init__(
        self,
        bond_types: Optional[Sequence[BondType]] = None,
        stereos: Optional[Sequence[int]] = None,
    ):
        self.bond_types = bond_types or [
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
        ]
        self.stereo = stereos or range(6)

    def __len__(self):
        return 4 + len(self.bond_types) + len(self.stereo)

    @property
    def subfeatures(self) -> dict[str, int]:
        names = ("null", "bond_type", "conjugated", "ring", "stereo")
        subfeature_sizes = [1, len(self.bond_types), 1, 1, (len(self.stereo) + 1)]
        offsets = np.cumsum([0] + subfeature_sizes[:-1])
        slices = [slice(i, j) for i, j in zip(offsets, offsets[1:])]

        return dict(zip(names, slices))

    def featurize(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)

        if b is None:
            x[0] = 1
            return x

        i = 1
        bond_type = b.GetBondType()
        bt_bit, size = self.one_hot_index(bond_type, self.bond_types)
        if bt_bit != size:
            x[i + bt_bit] = 1
        i += size - 1

        x[i] = int(b.GetIsConjugated())
        x[i + 1] = int(b.IsInRing())
        i += 2

        stereo_bit, _ = self.one_hot_index(int(b.GetStereo()), self.stereo)
        x[i + stereo_bit] = 1

        return x
