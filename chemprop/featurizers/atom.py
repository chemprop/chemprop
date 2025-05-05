from enum import auto
from typing import Sequence

import numpy as np
from rdkit.Chem.rdchem import Atom, ChiralType, HybridizationType

from chemprop.featurizers.base import MultiHotFeaturizer, OneHotFeaturizer, ValueFeaturizer
from chemprop.utils.utils import EnumMapping


class MultiHotAtomFeaturizer(MultiHotFeaturizer[Atom]):
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
        the choices for the total number of bonds an atom is engaged in.
    formal_charges : Sequence[int]
        the choices for integer electronic charge assigned to an atom.
    chiral_tags : Sequence[ChiralType | int]
        the choices for an atom's chiral tag. See `ChiralType`_ for possible integer values.
    num_Hs : Sequence[int]
        the choices for the total number of bonded hydrogen atoms (explicit and implicit).
    hybridizations : Sequence[HybridizationType | int]
        the choices for an atom’s hybridization type. See `HybridizationType`_ for possible integer values.

        .. _ChiralType: https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.ChiralType
        .. _HybridizationType: https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.HybridizationType

    Example
    -------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C[C@H](O)c1ccccc1")
    >>> featurizer = MultiHotAtomFeaturizer.v2()
    >>> for index in range(4):
    ...     print(featurizer.to_string(mol.GetAtomWithIdx(index)))
    00000100000000000000000000000000000000 0000100 000010 10000 000100 00001000 0 0.120
    00000100000000000000000000000000000000 0000100 000010 00100 010000 00001000 0 0.120
    00000001000000000000000000000000000000 0010000 000010 10000 010000 00001000 0 0.160
    00000100000000000000000000000000000000 0001000 000010 10000 100000 00100000 1 0.120

    """

    def __init__(
        self,
        atomic_nums: Sequence[int],
        degrees: Sequence[int],
        formal_charges: Sequence[int],
        chiral_tags: Sequence[ChiralType | int],
        num_Hs: Sequence[int],
        hybridizations: Sequence[HybridizationType | int],
    ):
        self.atomic_nums = atomic_nums
        self.degrees = degrees
        self.formal_charges = formal_charges
        self.chiral_tags = chiral_tags
        self.num_Hs = num_Hs
        self.hybridizations = hybridizations

        super().__init__(
            OneHotFeaturizer(lambda a: a.GetAtomicNum(), atomic_nums, padding=True),
            OneHotFeaturizer(lambda a: a.GetTotalDegree(), degrees, padding=True),
            OneHotFeaturizer(lambda a: a.GetFormalCharge(), formal_charges, padding=True),
            OneHotFeaturizer(lambda a: a.GetChiralTag(), chiral_tags, padding=True),
            OneHotFeaturizer(lambda a: a.GetTotalNumHs(), num_Hs, padding=True),
            OneHotFeaturizer(lambda a: a.GetHybridization(), hybridizations, padding=True),
            ValueFeaturizer(lambda a: a.GetIsAromatic(), int),
            ValueFeaturizer(lambda a: 0.01 * a.GetMass(), float),
        )

    def num_only(self, a: Atom | None) -> np.ndarray:
        """Featurize the atom by setting only the atomic number bit

        Parameters
        ----------
        a : Atom
            the atom to featurize

        Returns
        -------
        np.ndarray
            the featurized atom

        Example
        -------
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("C[C@H](O)c1ccccc1")
        >>> atom = mol.GetAtomWithIdx(0)
        >>> featurizer = MultiHotAtomFeaturizer.v1()
        >>> vector = featurizer.num_only(atom)
        >>> vector[atom.GetAtomicNum() - 1]
        1.0
        >>> sum(vector)
        1.0
        >>> len(vector) == len(featurizer)
        True

        """
        num_subfeat = self.subfeats[0]
        return np.concatenate([num_subfeat(a), np.zeros(len(self) - len(num_subfeat))])

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
            chiral_tags=[
                ChiralType.CHI_UNSPECIFIED,
                ChiralType.CHI_TETRAHEDRAL_CW,
                ChiralType.CHI_TETRAHEDRAL_CCW,
                ChiralType.CHI_OTHER,
            ],
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
            atomic_nums=[*range(1, 37), 53],
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            chiral_tags=[
                ChiralType.CHI_UNSPECIFIED,
                ChiralType.CHI_TETRAHEDRAL_CW,
                ChiralType.CHI_TETRAHEDRAL_CCW,
                ChiralType.CHI_OTHER,
            ],
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
            chiral_tags=[
                ChiralType.CHI_UNSPECIFIED,
                ChiralType.CHI_TETRAHEDRAL_CW,
                ChiralType.CHI_TETRAHEDRAL_CCW,
                ChiralType.CHI_OTHER,
            ],
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP3,
            ],
        )


class RIGRAtomFeaturizer(MultiHotFeaturizer[Atom]):
    """A :class:`RIGRAtomFeaturizer` uses a multi-hot encoding to featurize atoms using resonance-invariant features.

    The generated atom features are ordered as follows:
    * atomic number
    * degree
    * number of hydrogens
    * mass

    Parameters
    ----------
    atomic_nums : Sequence[int]
        the choices for atom type denoted by atomic number. Ex: ``[4, 5, 6]`` for C, N and O.
    degrees : Sequence[int]
        the choices for the total number of bonds an atom is engaged in.
    num_Hs : Sequence[int]
        the choices for the total number of bonded hydrogen atoms (explicit and implicit).

    Example
    -------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C[C@H](O)c1ccccc1")
    >>> featurizer = RIGRAtomFeaturizer()
    >>> for index in range(4):
    ...     print(featurizer.to_string(mol.GetAtomWithIdx(index)))
    00000100000000000000000000000000000000 0000100 000100 0.120
    00000100000000000000000000000000000000 0000100 010000 0.120
    00000001000000000000000000000000000000 0010000 010000 0.160
    00000100000000000000000000000000000000 0001000 100000 0.120

    """

    def __init__(
        self,
        atomic_nums: Sequence[int] | None = None,
        degrees: Sequence[int] | None = None,
        num_Hs: Sequence[int] | None = None,
    ):
        self.atomic_nums = atomic_nums or (*range(1, 37), 53)
        self.degrees = degrees or tuple(range(6))
        self.num_Hs = num_Hs or tuple(range(5))

        super().__init__(
            OneHotFeaturizer(lambda a: a.GetAtomicNum(), self.atomic_nums, padding=True),
            OneHotFeaturizer(lambda a: a.GetTotalDegree(), self.degrees, padding=True),
            OneHotFeaturizer(lambda a: a.GetTotalNumHs(), self.num_Hs, padding=True),
            ValueFeaturizer(lambda a: 0.01 * a.GetMass(), float),
        )


class AtomFeatureMode(EnumMapping):
    """The mode of an atom is used for featurization into a `MolGraph`"""

    V1 = auto()
    V2 = auto()
    ORGANIC = auto()
    RIGR = auto()


def get_multi_hot_atom_featurizer(mode: str | AtomFeatureMode) -> MultiHotAtomFeaturizer:
    """Build the corresponding multi-hot atom featurizer."""
    match AtomFeatureMode.get(mode):
        case AtomFeatureMode.V1:
            return MultiHotAtomFeaturizer.v1()
        case AtomFeatureMode.V2:
            return MultiHotAtomFeaturizer.v2()
        case AtomFeatureMode.ORGANIC:
            return MultiHotAtomFeaturizer.organic()
        case AtomFeatureMode.RIGR:
            return RIGRAtomFeaturizer()
        case _:
            raise RuntimeError("unreachable code reached!")
