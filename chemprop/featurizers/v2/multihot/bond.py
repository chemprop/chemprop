from typing import Optional, Sequence

import numpy as np
from rdkit.Chem.rdchem import Bond, BondType

from chemprop.featurizers.multihot.base import MultiHotFeaturizer


class BondFeaturizer(MultiHotFeaturizer):
    def __init__(
        self,
        bond_types: Optional[Sequence[BondType]] = None,
        stereo: Optional[Sequence[int]] = None,
    ):
        self.bond_types = bond_types or [
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
        ]
        self.stereo = stereo or range(6)

        self.subfeature_sizes = [1, len(self.bond_types), 1, 1, (len(self.stereo) + 1)]
        self.__size = sum(self.subfeature_sizes)

        names = ("null", "bond_type", "conjugated", "ring", "stereo")
        self.offsets = np.cumsum([0] + self.subfeature_sizes[:-1])
        self.__subfeatures = dict(zip(names, self.offsets))

        self.null_bit = 0
        self.bt_offset = 1
        self.conj_bit = self.bt_offset + len(self.bond_types)
        self.ring_bit = self.conj_bit + 1
        self.stero_offset = self.ring_bit + 1

        super().__init__()

    def __len__(self):
        return self.__size

    @property
    def subfeatures(self) -> dict[str, int]:
        return self.__subfeatures

    def featurize(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)

        if b is None:
            x[0] = 1
            return x

        bond_type = b.GetBondType()
        bt_bit = self.safe_index(bond_type, self.bond_types)
        if bt_bit != -1:
            x[self.bt_offset :][bt_bit] = 1

        x[self.conj_bit] = int(b.GetIsConjugated())
        x[self.ring_bit] = int(b.IsInRing())

        stereo_bit = self.safe_index(int(b.GetStereo()), self.stereo)
        x[self.stero_offset :][stereo_bit] = 1

        return x
