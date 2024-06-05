from enum import auto
from typing import Sequence

import numpy as np
from rdkit.Chem.rdchem import Atom, HybridizationType

from chemprop.featurizers.base import VectorFeaturizer
from chemprop.utils.utils import EnumMapping


class MultiHotAtomFeaturizer(VectorFeaturizer[Atom]):
    """A :class:`MultiHotAtomFeaturizer` uses a multi-hot encoding to featurize atoms.

    .. seealso::
        The class provides three default parameterization schemes:

        * :meth:`MultiHotAtomFeaturizer.v1`
        * :meth:`MultiHotAtomFeaturizer.v2`
        * :meth:`MultiHotAtomFeaturizer.organic`

    The generated atom features are ordered as follows:
    * atomic number
    * degree
    * formal charge
    * chiral tag
    * number of hydrogens
    * hybridization
    * aromaticity
    * mass

    .. important::
        Each feature, except for aromaticity and mass, includes a pad for unknown values.

    Parameters
    ----------
    atomic_nums : Sequence[int]
        the choices for atom type denoted by atomic number. Ex: ``[4, 5, 6]`` for C, N and O.
    degrees : Sequence[int]
        the choices for number of bonds an atom is engaged in.
    formal_charges : Sequence[int]
        the choices for integer electronic charge assigned to an atom.
    chiral_tags : Sequence[int]
        the choices for an atom's chiral tag. See :class:`rdkit.Chem.rdchem.ChiralType` for possible integer values.
    num_Hs : Sequence[int]
        the choices for number of bonded hydrogen atoms.
    hybridizations : Sequence[int]
        the choices for an atom’s hybridization type. See :class:`rdkit.Chem.rdchem.HybridizationType` for possible integer values.
    """

    def __init__(
        self,
        atomic_nums: Sequence[int],
        degrees: Sequence[int],
        formal_charges: Sequence[int],
        chiral_tags: Sequence[int],
        num_Hs: Sequence[int],
        hybridizations: Sequence[int],
    ):
        self.atomic_nums = {j: i for i, j in enumerate(atomic_nums)}
        self.degrees = {i: i for i in degrees}
        self.formal_charges = {j: i for i, j in enumerate(formal_charges)}
        self.chiral_tags = {i: i for i in chiral_tags}
        self.num_Hs = {i: i for i in num_Hs}
        self.hybridizations = {ht: i for i, ht in enumerate(hybridizations)}

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
    def v1(cls, max_atomic_num: int = 100):
        """The original implementation used in Chemprop V1 [1]_, [2]_.

        Parameters
        ----------
        max_atomic_num : int, default=100
            Include a bit for all atomic numbers in the interval :math:`[1, \mathtt{max\_atomic\_num}]`

        References
        -----------
        .. [1] Yang, K.; Swanson, K.; Jin, W.; Coley, C.; Eiden, P.; Gao, H.; Guzman-Perez, A.; Hopper, T.;
            Kelley, B.; Mathea, M.; Palmer, A. "Analyzing Learned Molecular Representations for Property Prediction."
            J. Chem. Inf. Model. 2019, 59 (8), 3370–3388. https://doi.org/10.1021/acs.jcim.9b00237
        .. [2] Heid, E.; Greenman, K.P.; Chung, Y.; Li, S.C.; Graff, D.E.; Vermeire, F.H.; Wu, H.; Green, W.H.; McGill,
            C.J. "Chemprop: A machine learning package for chemical property prediction." J. Chem. Inf. Model. 2024,
            64 (1), 9–17. https://doi.org/10.1021/acs.jcim.3c01250
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
    def v2(cls):
        """An implementation that includes an atom type bit for all elements in the first four rows of the periodic table plus iodine."""

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
    def organic(cls):
        r"""A specific parameterization intended for use with organic or drug-like molecules.

        This parameterization features:
            1. includes an atomic number bit only for H, B, C, N, O, F, Si, P, S, Cl, Br, and I atoms
            2. a hybridization bit for :math:`s, sp, sp^2` and :math:`sp^3` hybridizations.
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


class AtomFeatureMode(EnumMapping):
    """The mode of an atom is used for featurization into a `MolGraph`"""

    V1 = auto()
    V2 = auto()
    ORGANIC = auto()


def get_multi_hot_atom_featurizer(mode: str | AtomFeatureMode) -> MultiHotAtomFeaturizer:
    """Build the corresponding multi-hot atom featurizer."""
    match AtomFeatureMode.get(mode):
        case AtomFeatureMode.V1:
            return MultiHotAtomFeaturizer.v1()
        case AtomFeatureMode.V2:
            return MultiHotAtomFeaturizer.v2()
        case AtomFeatureMode.ORGANIC:
            return MultiHotAtomFeaturizer.organic()
        case _:
            raise RuntimeError("unreachable code reached!")
