from dataclasses import InitVar, dataclass, field
from typing import Protocol, Sequence

import numpy as np
from rdkit.Chem.rdchem import Atom, HybridizationType

from chemprop.v2.featurizers.protos import AtomFeaturizerProto


@dataclass(repr=False, eq=False, slots=True)
class AtomFeaturizer(AtomFeaturizerProto):
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
    | 0-101               | atomic number   | Y            |
    +---------------------+-----------------+--------------+
    | 101-108             | degree          | Y            |
    +---------------------+-----------------+--------------+
    | 108-114             | formal charge   | Y            |
    +---------------------+-----------------+--------------+
    | 114-119             | chiral tag      | Y            |
    +---------------------+-----------------+--------------+
    | 119-125             | # Hs            | Y            |
    +---------------------+-----------------+--------------+
    | 125-131             | hybridization   | Y            |
    +---------------------+-----------------+--------------+
    | 131-132             | aromatic?       | N            |
    +---------------------+-----------------+--------------+
    | 132-133             | mass            | N            |
    +---------------------+-----------------+--------------+

    NOTE: the above signature only applies for the default arguments, as the each slice (save for
    the final two) can increase in size depending on the input arguments.
    """

    max_atomic_num: InitVar[int] = 100
    degrees: Sequence[int] = field(default_factory=lambda: range(6))
    formal_charges: Sequence[int] = field(default_factory=lambda: [-1, -2, 1, 2, 0])
    chiral_tags: Sequence[int] = field(default_factory=lambda: range(4))
    num_Hs: Sequence[int] = field(default_factory=lambda: range(5))
    hybridizations: Sequence[HybridizationType] = field(
        default_factory=lambda: [
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2,
        ]
    )

    def __post_init__(self, max_atomic_num: int = 100):
        self.atomic_nums = {i: i for i in range(max_atomic_num)}
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
            a.GetAtomicNum() - 1,
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
