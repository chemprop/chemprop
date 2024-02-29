from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import Sequence

import numpy as np
from rdkit.Chem.rdchem import Atom, HybridizationType


class AtomFeaturizer(ABC):
    """An :class:`AtomFeaturizer` calculates feature vectors of RDKit atoms."""

    @abstractmethod
    def __len__(self) -> int:
        """the length of an atomic feature vector"""

    @abstractmethod
    def __call__(self, a: Atom) -> np.ndarray:
        """featurize the atom ``a``"""


@dataclass(repr=False, eq=False, slots=True)
class MultiHotAtomFeaturizer(AtomFeaturizer):
    """An :class:`AtomFeaturizer` featurizes atoms based on the following attributes:

    * atomic number
    * degree
    * formal charge
    * chiral tag
    * number of hydrogens
    * hybridization
    * aromaticity
    * mass

    The feature vectors produced by this featurizer have the following (general) signature:

    +---------------------+-----------------+--------------+
    | slice [start, stop) | subfeature      | unknown pad? |
    +=====================+=================+==============+
    | 0-38                | atomic number   | Y            |
    +---------------------+-----------------+--------------+
    | 38-45               | degree          | Y            |
    +---------------------+-----------------+--------------+
    | 45-51               | formal charge   | Y            |
    +---------------------+-----------------+--------------+
    | 51-56               | chiral tag      | Y            |
    +---------------------+-----------------+--------------+
    | 56-62               | # Hs            | Y            |
    +---------------------+-----------------+--------------+
    | 62-71               | hybridization   | Y            |
    +---------------------+-----------------+--------------+
    | 71-72               | aromatic?       | N            |
    +---------------------+-----------------+--------------+
    | 72-73               | mass            | N            |
    +---------------------+-----------------+--------------+

    NOTE: the above signature only applies for the default arguments, as the each slice (save for
    the final two) can increase in size depending on the input arguments.
    """

    # all elements in the first 4 rows of periodic talbe plus iodine and 0 padding for other elements
    atomic_nums: Sequence[int] = field(default_factory=lambda: list(range(1, 37)) + [53])
    degrees: Sequence[int] = field(default_factory=lambda: range(6))
    formal_charges: Sequence[int] = field(default_factory=lambda: [-1, -2, 1, 2, 0])
    chiral_tags: Sequence[int] = field(default_factory=lambda: range(4))
    num_Hs: Sequence[int] = field(default_factory=lambda: range(5))
    hybridizations: Sequence[HybridizationType] = field(
        default_factory=lambda: [
            HybridizationType.S,
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP2D,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2,
            HybridizationType.OTHER,
        ]
    )

    def __post_init__(self):
        self.atomic_nums = {j: i for i, j in enumerate(self.atomic_nums)}
        self.degrees = {i: i for i in self.degrees}
        self.formal_charges = {j: i for i, j in enumerate(self.formal_charges)}
        self.chiral_tags = {i: i for i in self.chiral_tags}
        self.num_Hs = {i: i for i in self.num_Hs}
        self.hybridizations = {ht: i for i, ht in enumerate(self.hybridizations)}
        self._subfeats: list[dict] = [
            self.atomic_nums,
            self.degrees,
            self.formal_charges,
            self.chiral_tags,
            self.num_Hs,
            self.hybridizations,
        ]
        subfeat_sizes = [
            1 + len(self.atomic_nums),
            1 + len(self.degrees),
            1 + len(self.formal_charges),
            1 + len(self.chiral_tags),
            1 + len(self.num_Hs),
            1 + len(self.hybridizations),
            1,
            1,
        ]
        self.__size = sum(subfeat_sizes)

    def __len__(self) -> int:
        return self.__size

    def __call__(self, a: Atom | None) -> np.ndarray:
        x = np.zeros(self.__size)

        if a is None:
            return x

        feats = [
            a.GetAtomicNum(),
            a.GetTotalDegree(),
            a.GetFormalCharge(),
            int(a.GetChiralTag()),
            int(a.GetTotalNumHs()),
            a.GetHybridization(),
        ]
        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()

        return x

    def num_only(self, a: Atom) -> np.ndarray:
        """featurize the atom by setting only the atomic number bit"""
        x = np.zeros(len(self))

        if a is None:
            return x

        i = self.atomic_nums.get(a.GetAtomicNum() - 1, len(self.atomic_nums))
        x[i] = 1

        return x
