from dataclasses import InitVar, dataclass, field, fields
from typing import Optional, Sequence

import numpy as np
from rdkit.Chem.rdchem import Atom, HybridizationType

from chemprop.featurizers.utils import safe_index

class AtomFeaturizer:
    def __init__(
        self,
        max_atomic_num: int = 100,
        degree: Optional[Sequence[int]] = None,
        formal_charge: Optional[Sequence[int]] = None,
        chiral_tag: Optional[Sequence[int]] = None,
        num_Hs: Optional[Sequence[int]] = None,
        hybridization: Optional[Sequence[HybridizationType]] = None
    ):
        self.atomic_num = list(range(max_atomic_num))
        self.degree = degree or list(range(6))
        self.formal_charge = formal_charge or [-1, -2, 1, 2, 0]
        self.chiral_tag = chiral_tag or list(range(4))
        self.num_Hs = num_Hs or list(range(5))
        self.hybridization = hybridization or [
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2
        ]

        self.__length = sum(
            len(features) + 1
            for features in (
                self.atomic_num,
                self.degree,
                self.formal_charge,
                self.chiral_tag,
                self.num_Hs,
                self.hybridization
            )
        ) + 2

    def __len__(self):
        """the dimension of an atom feature vector, adding 1 to each set of features for uncommon
        values and 2 at the end to account for aromaticity and mass"""
        return self.__length

    def __call__(self, a: Atom) -> np.ndarray:
        return self.featurize(a)

    def featurize(self, a: Atom) -> np.ndarray:
        x = np.zeros(len(self))

        if a is None:
            return x

        bits_offsets = [
            safe_index((a.GetAtomicNum() -1), self.atomic_num),
            safe_index(a.GetTotalDegree(), self.degree),
            safe_index(a.GetFormalCharge(), self.formal_charge),
            safe_index(int(a.GetChiralTag()), self.chiral_tag),
            safe_index(int(a.GetTotalNumHs()), self.num_Hs),
            safe_index(int(a.GetHybridization()), self.hybridization),
        ]

        i = 0
        for bit, offset in bits_offsets:
            x[i + bit] = 1
            i += offset
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()
        
        return x