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
    """An :class:`AtomFeaturizer` featurizes atoms using a multi-hot encoding scheme. Having specific implementations to customize certain features.
    
    The Chemprop default atom features are:
    
    * atomic number
    * degree
    * formal charge
    * chiral tag
    * number of hydrogens
    * hybridization
    * aromaticity
    * mass
    """
    
    atomic_nums: Sequence[int] = field(default_factory=list)
    degrees: Sequence[int] = field(default_factory=list)
    formal_charges: Sequence[int] = field(default_factory=list)
    chiral_tags: Sequence[int] = field(default_factory=list)
    num_Hs: Sequence[int] = field(default_factory=list)
    hybridizations: Sequence[HybridizationType] = field(default_factory=list)
    
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
    
    @classmethod
    def default(cls):
        """A specific implementation of MultiHotAtomFeaturizer that includes features only for atoms in common molecules. 
        Includes all elements in the first 4 rows of the periodic talbe plus iodine and an 0 padding for other elements.
        This is the current default in Chemprop. 
        
        The feature vectors produced by this featurizer have the following signature:

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
        """

        return cls(atomic_nums = list(range(1, 37)) + [53],
                   degrees = list(range(6)),
                   formal_charges = [-1, -2, 1, 2, 0],
                   chiral_tags = list(range(4)),
                   num_Hs = list(range(5)),
                   hybridizations = [
                       HybridizationType.S,
                       HybridizationType.SP,
                       HybridizationType.SP2,
                       HybridizationType.SP2D,
                       HybridizationType.SP3,
                       HybridizationType.SP3D,
                       HybridizationType.SP3D2,
                       HybridizationType.OTHER,
                   ])

    @classmethod
    def v1(cls, MAX_ATOMIC_NUM = 100):
        """A specific implementation of MultiHotAtomFeaturizer that corresponds to the default in Chemprop v1. Ref. [1] & [2].
        
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
        
        Parameters
        ----------
        MAX_ATOMIC_NUM : int, default 100
            The maximum atomic number to include in the feature vector. The default is 100.
            
        References
        ----------
        [1] J. Chem. Inf. Model. 2019, 59, 8, 3370–3388
        [2] J. Chem. Inf. Model. 2024, 64, 1, 9–17
        """

        return cls(atomic_nums = list(range(MAX_ATOMIC_NUM)),
                   degrees = list(range(6)),
                   formal_charges = [-1, -2, 1, 2, 0],
                   chiral_tags = list(range(4)),
                   num_Hs = list(range(5)),
                   hybridizations = [
                       HybridizationType.SP,
                       HybridizationType.SP2,
                       HybridizationType.SP3,
                       HybridizationType.SP3D,
                       HybridizationType.SP3D2,
                   ])
        
    @classmethod
    def organic(cls):
        """A specific implementation of MultiHotAtomFeaturizer that includes features only for atoms in common organic molecules. 
        Includes H, B, C, N, O, F, Si, P, S, Cl, Br, I and an 0 padding for other elements.
        Intended for use with organic molecules for drug research and development.
        
        The feature vectors produced by this featurizer have the following signature:

        +---------------------+-----------------+--------------+
        | slice [start, stop) | subfeature      | unknown pad? |
        +=====================+=================+==============+
        | 0-13                | atomic number   | Y            |
        +---------------------+-----------------+--------------+
        | 13-20               | degree          | Y            |
        +---------------------+-----------------+--------------+
        | 20-26               | formal charge   | Y            |
        +---------------------+-----------------+--------------+
        | 26-31               | chiral tag      | Y            |
        +---------------------+-----------------+--------------+
        | 31-37               | # Hs            | Y            |
        +---------------------+-----------------+--------------+
        | 37-44               | hybridization   | Y            |
        +---------------------+-----------------+--------------+
        | 44-45               | aromatic?       | N            |
        +---------------------+-----------------+--------------+
        | 45-46               | mass            | N            |
        +---------------------+-----------------+--------------+
        """
        
        return cls(atomic_nums = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53],
                   degrees = list(range(6)),
                   formal_charges = [-1, -2, 1, 2, 0],
                   chiral_tags = list(range(4)),
                   num_Hs = list(range(5)),
                   hybridizations = [
                       HybridizationType.S,
                       HybridizationType.SP,
                       HybridizationType.SP2,
                       HybridizationType.SP3,
                       HybridizationType.SP3D,
                       HybridizationType.SP3D2,
                   ])
