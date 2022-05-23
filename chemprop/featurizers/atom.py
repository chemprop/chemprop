from dataclasses import InitVar, dataclass, field, fields
from typing import Sequence

import numpy as np
from rdkit.Chem.rdchem import Atom, HybridizationType


def safe_index(x, xs: Sequence):
    """return both the index of `x` in `xs` (if it exists, else -1) and the total length of `xs`"""
    return xs.index(x) if x in xs else len(xs), len(xs)


@dataclass(frozen=True)
class AtomFeaturizer:
    max_atomic_num: InitVar[int] = 100
    atomic_num: list[int] = field(init=False)
    degree: list[int] = field(default_factory=lambda: list(range(6)))
    formal_charge: list[int] = field(default_factory=lambda: [-1, -2, 1, 2, 0])
    chiral_tag: list[int] = field(default_factory=lambda: list(range(4)))
    num_Hs: list[int] = field(default_factory=lambda: list(range(5)))
    hybridization: list[HybridizationType] = field(
        default_factory=lambda: [
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2
        ]
    )

    def __post_init__(self, max_atomic_num: int):
        self.atomic_num = list(range(max_atomic_num))
        self.__length = sum(len(getattr(self, field.name)) + 1 for field in fields(self)) + 2

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
        x[i + 1] = [0.01 * a.Getmass()]
        
        return x