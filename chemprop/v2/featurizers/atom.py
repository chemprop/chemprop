from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
from rdkit.Chem.rdchem import Atom, HybridizationType

from chemprop.v2.featurizers.utils import MultiHotFeaturizerMixin


class AtomFeaturizerBase(ABC):
    """An `AtomFeaturizerBase` calculates feature vectors of RDKit atoms."""

    @abstractmethod
    def __len__(self) -> int:
        """the length of an atomic feature vector"""

    @abstractmethod
    def __call__(self, a: Atom) -> np.ndarray:
        """featurize the atom `a`"""


class AtomFeaturizer(MultiHotFeaturizerMixin, AtomFeaturizerBase):
    """An `AtomFeaturizer` calculates feature vectors of RDKit atoms.

    The featurizations produced by this featurizer have the following (general) signature:

    | slice   | subfeature      | unknown pad? |
    | ------- | --------------- | ------------ |
    | 0-101   | atomic number   | Y            |
    | 101-108 | degree          | Y            |
    | 108-114 | formal charge   | Y            |
    | 114-119 | chiral tag      | Y            |
    | 119-125 | # Hs            | Y            |
    | 125-131 | hybridization   | Y            |
    | 131-132 | aromatic?       | N            |
    | 132-133 | mass            | N            |

    NOTE: the above signature only applies for the default arguments, as the each slice (save for
    the final two) can increase in size depending on the input arguments.

    Parameters
    ----------
    max_atomic_num : int, default=100
        the maximum atomic number categorized, by
    degrees : Optional[Sequence[int]], default=[0, 1, 2, 3, 4, 5]
        the categories for the atomic degree
    formal_charges : Optional[Sequence[int]], default=[-1, -2, 1, 2, 0]
        the categories for formal charge of an atom
    chiral_tags : Optional[Sequence[int]], default=[0, 1, 2, 3]
        the categories for the chirality of an atom
    num_Hs : Optional[Sequence[int]], default=[0, 1, 2, 3, 4]
        the categories for the number of hydrogens attached to an atom
    hybridizations : Optional[Sequence[HybridizationType]], default=[SP, SP2, SP3, SP3D, SP3D2]
        the categories for the hybridization of an atom
    """

    def __init__(
        self,
        max_atomic_num: int = 100,
        degrees: Optional[Sequence[int]] = None,
        formal_charges: Optional[Sequence[int]] = None,
        chiral_tags: Optional[Sequence[int]] = None,
        num_Hs: Optional[Sequence[int]] = None,
        hybridizations: Optional[Sequence[HybridizationType]] = None,
    ):
        self.max_atomic_num = max_atomic_num
        self.atomic_nums = range(max_atomic_num)
        self.degrees = degrees or range(6)
        self.formal_charges = formal_charges or [-1, -2, 1, 2, 0]
        self.chiral_tags = chiral_tags or range(4)
        self.num_Hs = num_Hs or range(5)
        self.hybridizations = hybridizations or [
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2,
        ]

    def __len__(self) -> int:
        return (
            len(self.atomic_nums)
            + 1
            + len(self.degrees)
            + 1
            + len(self.formal_charges)
            + 1
            + len(self.chiral_tags)
            + 1
            + len(self.num_Hs)
            + 1
            + len(self.hybridizations)
            + 1
            + 2
        )

    @property
    def choicess(self) -> list[Sequence]:
        return [
            self.atomic_nums,
            self.degrees,
            self.formal_charges,
            self.chiral_tags,
            self.num_Hs,
            self.hybridizations,
        ]

    @property
    def subfeatures(self) -> list[str, slice]:
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
        sizes = [len(choices) + 1 for choices in self.choicess] + [1, 1]
        offsets = np.cumsum([0] + sizes)
        slices = [slice(i, j) for i, j in zip(offsets, offsets[1:])]

        return dict(zip(names, slices))

    def featurize(self, a: Atom) -> np.ndarray:
        x = np.zeros(len(self))

        if a is None:
            return x

        bits_sizes = [
            self.one_hot_index((a.GetAtomicNum() - 1), self.atomic_nums),
            self.one_hot_index(a.GetTotalDegree(), self.degrees),
            self.one_hot_index(a.GetFormalCharge(), self.formal_charges),
            self.one_hot_index(int(a.GetChiralTag()), self.chiral_tags),
            self.one_hot_index(int(a.GetTotalNumHs()), self.num_Hs),
            self.one_hot_index(int(a.GetHybridization()), self.hybridizations),
        ]

        i = 0
        for j, size in bits_sizes:
            x[i + j] = 1
            i += size
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()

        return x

    def featurize_num_only(self, a: Atom) -> np.ndarray:
        """featurize the atom and only set the atomic number bit"""
        x = np.zeros(len(self))

        if a is None:
            return x

        bit, _ = self.one_hot_index((a.GetAtomicNum() - 1), self.atomic_nums)
        x[bit] = 1

        return x
