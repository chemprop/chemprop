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
    """An :class:`MultiHotAtomFeaturizer` uses a multi-hot encoding to featurize atoms.
    
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
        self._atomic_nums = {j: i for i, j in enumerate(self.atomic_nums)}
        self._degrees = {i: i for i in self.degrees}
        self._formal_charges = {j: i for i, j in enumerate(self.formal_charges)}
        self._chiral_tags = {i: i for i in self.chiral_tags}
        self._num_Hs = {i: i for i in self.num_Hs}
        self._hybridizations = {ht: i for i, ht in enumerate(self.hybridizations)}
        self._subfeats: list[dict] = [
            self._atomic_nums,
            self._degrees,
            self._formal_charges,
            self._chiral_tags,
            self._num_Hs,
            self._hybridizations,
        ]
        subfeat_sizes = [
            1 + len(self._atomic_nums),
            1 + len(self._degrees),
            1 + len(self._formal_charges),
            1 + len(self._chiral_tags),
            1 + len(self._num_Hs),
            1 + len(self._hybridizations),
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

        i = self._atomic_nums.get(a.GetAtomicNum(), len(self._atomic_nums))
        x[i] = 1

        return x
    
    @classmethod
    def default(cls):
        """An implementation that includes features only for atoms in common molecules. 
        Includes all elements in the first four rows of the periodic table plus iodine. This is the default in Chemprop V2. 
        
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
        | 62-70               | hybridization   | Y            |
        +---------------------+-----------------+--------------+
        | 70-71               | aromatic?       | N            |
        +---------------------+-----------------+--------------+
        | 71-72               | mass            | N            |
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
                   ])

    @classmethod
    def v1(cls, max_atomic_num: int = 100):
        """The original implementation used in Chemprop V1 [1]_, [2]_.
        
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

        .. note:: 
        the above signature only applies for the default arguments, as the each slice (save for
        the final two) can increase in size depending on the input arguments.
        
        Parameters
        ----------
        max_atomic_num : int, default 100
            The maximum atomic number to include in the feature vector. The default is 100.
            
        References
        ----------
        .. [1] J. Chem. Inf. Model. 2019, 59, 8, 3370–3388
        .. [2] J. Chem. Inf. Model. 2024, 64, 1, 9–17
        """

        return cls(atomic_nums = list(range(1, max_atomic_num + 1)),
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
        | 37-43               | hybridization   | Y            |
        +---------------------+-----------------+--------------+
        | 43-44               | aromatic?       | N            |
        +---------------------+-----------------+--------------+
        | 44-45               | mass            | N            |
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
                   ])
