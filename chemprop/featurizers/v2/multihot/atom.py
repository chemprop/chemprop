from typing import Optional, Sequence

import numpy as np
from rdkit.Chem.rdchem import Atom, HybridizationType

from chemprop.featurizers.v2.multihot.base import MultiHotFeaturizer


class AtomFeaturizer(MultiHotFeaturizer):
    """An AtomFeaturizer calculates feature vectors of RDKit atoms.

    TODO: fix attribute assignment such that the class doesn't break immediately upon a change
    """

    def __init__(
        self,
        max_atomic_num: int = 100,
        degree: Optional[Sequence[int]] = None,
        formal_charge: Optional[Sequence[int]] = None,
        chiral_tag: Optional[Sequence[int]] = None,
        num_Hs: Optional[Sequence[int]] = None,
        hybridization: Optional[Sequence[HybridizationType]] = None,
    ):
        self.max_atomic_num = max_atomic_num
        self.__atomic_num = range(max_atomic_num)
        self.__degree = degree or range(6)
        self.__formal_charge = formal_charge or [-1, -2, 1, 2, 0]
        self.__chiral_tag = chiral_tag or range(4)
        self.__num_Hs = num_Hs or range(5)
        self.__hybridization = hybridization or [
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2,
        ]

        self.subfeature_sizes = [
            len(features) + 1
            for features in (
                self.__atomic_num,
                self.__degree,
                self.__formal_charge,
                self.__chiral_tag,
                self.__num_Hs,
                self.__hybridization,
            )
        ] + [1, 1]
        self.__size = sum(self.subfeature_sizes)

        offsets = np.cumsum([0] + self.subfeature_sizes)
        names = (
            "atomic_num",
            "degree",
            "formal_charge",
            "chiral_tag",
            "num_Hs",
            "hybridization",
            "aromatic",
            "mass",
        )
        self.__subfeatures = dict(zip(names, offsets))

        super().__init__()

    def __len__(self):
        """the dimension of an atom feature vector"""
        return self.__size

    @property
    def subfeatures(self) -> dict[str, slice]:
        return self.__subfeatures

    def featurize(self, a: Atom) -> np.ndarray:
        x = np.zeros(len(self))

        if a is None:
            return x

        bits = [
            self.safe_index((a.GetAtomicNum() - 1), self.__atomic_num),
            self.safe_index(a.GetTotalDegree(), self.__degree),
            self.safe_index(a.GetFormalCharge(), self.__formal_charge),
            self.safe_index(int(a.GetChiralTag()), self.__chiral_tag),
            self.safe_index(int(a.GetTotalNumHs()), self.__num_Hs),
            self.safe_index(int(a.GetHybridization()), self.__hybridization),
        ]

        i = 0
        for j, size in zip(bits, self.subfeature_sizes):
            if j == -1:
                x[i + size - 1] = 1
            else:
                x[i + j] = 1
            i += size
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()

        return x

    def featurize_num_only(self, a: Atom) -> np.ndarray:
        x = np.zeros(len(self))

        if a is None:
            return x
        
        bit = self.safe_index((a.GetAtomicNum() - 1), self.__atomic_num)
        bit = bit if bit != -1 else self.max_atomic_num

        x[bit] = 1
        
        return x