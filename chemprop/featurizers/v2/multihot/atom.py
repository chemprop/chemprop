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
        self.atomic_num = range(max_atomic_num)
        self.degree = degree or range(6)
        self.formal_charge = formal_charge or [-1, -2, 1, 2, 0]
        self.chiral_tag = chiral_tag or range(4)
        self.num_Hs = num_Hs or range(5)
        self.hybridization = hybridization or [
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2,
        ]

        self.choicess = [
            self.atomic_num,
            self.degree,
            self.formal_charge,
            self.chiral_tag,
            self.num_Hs,
            self.hybridization,
        ]

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
        subfeature_sizes = [len(choices) + 1 for choices in self.choicess] + [1, 1]
        offsets = np.cumsum([0] + subfeature_sizes)
        slices = [slice(i, j) for i, j in zip(offsets, offsets[1:])]
        self.__subfeatures = dict(zip(names, slices))

        super().__init__()

    def __len__(self):
        return sum(len(xs) + 1 for xs in self.choicess) + 2

    @property
    def subfeatures(self) -> dict[str, slice]:
        return self.__subfeatures

    def featurize(self, a: Atom) -> np.ndarray:
        x = np.zeros(len(self))

        if a is None:
            return x

        bits_sizes = [
            self.one_hot_index((a.GetAtomicNum() - 1), self.atomic_num),
            self.one_hot_index(a.GetTotalDegree(), self.degree),
            self.one_hot_index(a.GetFormalCharge(), self.formal_charge),
            self.one_hot_index(int(a.GetChiralTag()), self.chiral_tag),
            self.one_hot_index(int(a.GetTotalNumHs()), self.num_Hs),
            self.one_hot_index(int(a.GetHybridization()), self.hybridization),
        ]

        i = 0
        for j, size in bits_sizes:
            x[i + j] = 1
            i += size
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()

        return x

    def featurize_num_only(self, a: Atom) -> np.ndarray:
        x = np.zeros(len(self))

        if a is None:
            return x

        bit = self.safe_index((a.GetAtomicNum() - 1), self.atomic_num)
        bit = bit if bit != -1 else self.max_atomic_num

        x[bit] = 1

        return x
