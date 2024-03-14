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

    This featurizer provides three configurations tailored to different chemical informatics needs:

    1. `default`: Tailored for a broad range of molecules, this configuration encompasses all elements
    in the first four rows of the periodic table, along with iodine. It is the default in Chemprop V2.
    2. `v1`: Corresponds to the original configuration employed in the Chemprop V1 [1]_, [2]_.
    3. `organic`: Designed specifically for use with organic molecules for drug research and development,
    this configuration includes a subset of elements most common in organic chemistry, including H, B, C, N, O, F, Si, P, S, Cl, Br, and I.

    
    Feature vector specifications for each configuration are detailed as follows:

    `default` configuration produces feature vectors with the following schema:

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


    `v1` configuration produces feature vectors with the following schema:

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

    **NOTE**: the above schema only applies for the default arguments, as the each slice (save for
    the final two) can increase in size depending on the input arguments.


    `organic` configuration produces feature vectors with the following schema:

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

    Parameters
    ----------
    atomic_nums : Sequence[int] | None, default=None
        type of atom (ex. C, N, O), by atomic number
    degree : Sequence[int] | None, default=None
        number of bonds the atom is involved in
    formal_charges : Sequence[int] | None, default=None
        integer electronic charge assigned to atom
    chiral_tags : Sequence[int] | None, default=None
        unspecified, tetrahedral CW/CCW, or other
    num_Hs : Sequence[int] | None, default=None
        number of bonded hydrogen atoms
    hybridizations : Sequence[int] | None, default=None
        type of atom’s hybridization (ex. sp, sp2, sp3, sp3d, or sp3d2)

    References
    ----------
    .. [1] Yang, K.; Swanson, K.; Jin, W.; Coley, C.; Eiden, P.; Gao, H.; Guzman-Perez, A.; Hopper, T.;
        Kelley, B.; Mathea, M.; Palmer, A. "Analyzing Learned Molecular Representations for Property Prediction."
        J. Chem. Inf. Model. 2019, 59, 8, 3370–3388. https://doi.org/10.1021/acs.jcim.9b00237
    .. [2] Heid, E.; Greenman, K.P.; Chung, Y.; Li, S.C., Graff, D.E.; Vermeire, F.H.; Wu, H.; Green, W.H.; McGill,
        C.J., 2023. "Chemprop: A machine learning package for chemical property prediction." J. Chem. Inf. Model. 2024,
        64, 1, 9–17. https://doi.org/10.1021/acs.jcim.3c01250
    """

    def __init__(
        self,
        atomic_nums: Sequence[int] | None = None,
        degrees: Sequence[int] | None = None,
        formal_charges: Sequence[int] | None = None,
        chiral_tags: Sequence[int] | None = None,
        num_Hs: Sequence[int] | None = None,
        hybridizations: Sequence[int] | None = None,
    ):
        if all(
            arg is None
            for arg in [atomic_nums, degrees, formal_charges, chiral_tags, num_Hs, hybridizations]
        ):
            # No custom parameters provided, use default settings
            default_settings = self.default()
            self.atomic_nums = default_settings.atomic_nums
            self.degrees = default_settings.degrees
            self.formal_charges = default_settings.formal_charges
            self.chiral_tags = default_settings.chiral_tags
            self.num_Hs = default_settings.num_Hs
            self.hybridizations = default_settings.hybridizations
        else:
            # Custom parameters provided, initialize accordingly
            self.atomic_nums = {j: i for i, j in enumerate(atomic_nums or [])}
            self.degrees = {i: i for i in degrees or []}
            self.formal_charges = {j: i for i, j in enumerate(formal_charges or [])}
            self.chiral_tags = {i: i for i in chiral_tags or []}
            self.num_Hs = {i: i for i in num_Hs or []}
            self.hybridizations = {ht: i for i, ht in enumerate(hybridizations or [])}

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

        i = self.atomic_nums.get(a.GetAtomicNum(), len(self.atomic_nums))
        x[i] = 1

        return x

    @classmethod
    def default(cls):
        """An implementation that includes features only for atoms in common molecules.
        Includes all elements in the first four rows of the periodic table plus iodine. This is the default in Chemprop V2.
        """

        return cls(
            atomic_nums=list(range(1, 37)) + [53],
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            chiral_tags=list(range(4)),
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP2D,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ],
        )

    @classmethod
    def v1(cls, max_atomic_num: int = 100):
        """The original implementation used in Chemprop V1.

        Parameters
        ----------
        max_atomic_num : int, default 100
            The maximum atomic number to include in the feature vector. The default is 100.
        """

        return cls(
            atomic_nums=list(range(1, max_atomic_num + 1)),
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            chiral_tags=list(range(4)),
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ],
        )

    @classmethod
    def organic(cls):
        """A specific implementation that includes features only for atoms in common organic molecules.
        Includes H, B, C, N, O, F, Si, P, S, Cl, Br, and I.
        Intended for use with organic molecules for drug research and development.
        """

        return cls(
            atomic_nums=[1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53],
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            chiral_tags=list(range(4)),
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP3,
            ],
        )
