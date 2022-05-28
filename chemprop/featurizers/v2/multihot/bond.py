from typing import Optional, Sequence

import numpy as np
from rdkit.Chem.rdchem import Bond, BondType

from chemprop.featurizers.v2.multihot.base import MultiHotFeaturizer


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

        names = ("null", "bond_type", "conjugated", "ring", "stereo")
        subfeature_sizes = [1, len(self.bond_types), 1, 1, (len(self.stereo) + 1)]
        offsets = np.cumsum([0] + subfeature_sizes[:-1])
        slices = [slice(i, j) for i, j in zip(offsets, offsets[1:])]
        self.__subfeatures = dict(zip(names, slices))

    def __len__(self):
        return 4 + len(self.bond_types) + len(self.stereo)

    @property
    def subfeatures(self) -> dict[str, int]:
        return self.__subfeatures

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
        i += (size -1)

        x[i] = int(b.GetIsConjugated())
        x[i + 1] = int(b.IsInRing())
        i += 2

        stereo_bit, _ = self.one_hot_index(int(b.GetStereo()), self.stereo)
        x[i + stereo_bit] = 1

        return x
