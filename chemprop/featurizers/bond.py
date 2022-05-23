from dataclasses import dataclass

import numpy as np
from rdkit.Chem.rdchem import Bond

@dataclass
class BondFeaturizer:
    bond_fdim: int = 14

    def __len__(self):
        return self.bond_fdim
        
    def __call__(self, b: Bond) -> np.ndarray:
        return self.featurize(b)

    def featurize(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self))
        
        if b is None:
            x[0] = 1
            return x

        bond_type = b.GetBondType()

        if bond_type is not None:
            bt_int = int(bond_type)
            CONJ_BIT = 5
            RING_BIT = 6

            if bt_int in {1, 2, 3, 12}:
                x[max(4, bt_int)] = 1
            if b.GetIsConjugated():
                x[CONJ_BIT] = 1
            if b.IsInRing():
                x[RING_BIT] = 1

        stereo_bit = int(b.GetStereo())
        x[stereo_bit] = 1

        return x