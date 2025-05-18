from typing import Sequence

from rdkit.Chem.rdchem import Bond, BondStereo, BondType

from chemprop.featurizers.base import (
    MultiHotFeaturizer,
    NullityFeaturizer,
    OneHotFeaturizer,
    ValueFeaturizer,
)


class MultiHotBondFeaturizer(MultiHotFeaturizer[Bond]):
    """A :class:`MultiHotBondFeaturizer` feauturizes bonds based on the following attributes:

    * ``null``-ity (i.e., is the bond ``None``?)
    * bond type
    * conjugated?
    * in ring?
    * stereochemistry

    The feature vectors produced by this featurizer have the following (general) signature:

    +---------------------+-----------------+--------------+
    | slice [start, stop) | subfeature      | unknown pad? |
    +=====================+=================+==============+
    | 0-1                 | null?           | N            |
    +---------------------+-----------------+--------------+
    | 1-5                 | bond type       | N            |
    +---------------------+-----------------+--------------+
    | 5-6                 | conjugated?     | N            |
    +---------------------+-----------------+--------------+
    | 6-8                 | in ring?        | N            |
    +---------------------+-----------------+--------------+
    | 7-14                | stereochemistry | Y            |
    +---------------------+-----------------+--------------+

    **NOTE**: the above signature only applies for the default arguments, as the bond type and
    sterochemistry slices can increase in size depending on the input arguments.

    Parameters
    ----------
    bond_types : Sequence[BondType], default=[SINGLE, DOUBLE, TRIPLE, AROMATIC]
        the known bond types
    stereos : Sequence[BondStereo], default=[NONE, ANY, Z, E, CIS, TRANS]
        the known bond stereochemistries. See [1]_ for more details

    Example
    -------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C=C-c1ccccc1')
    >>> featurizer = MultiHotBondFeaturizer()
    >>> for i in range(4):
    ...     print(featurizer.to_string(mol.GetBondWithIdx(i)))
    0 0100 1 0 1000000
    0 1000 1 0 1000000
    0 0001 1 1 1000000
    0 0001 1 1 1000000

    References
    ----------
    .. [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondStereo.values

    """

    def __init__(
        self,
        bond_types: Sequence[BondType] | None = None,
        stereos: Sequence[BondStereo] | None = None,
    ):
        self.bond_types = bond_types or (
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
        )
        self.stereos = stereos or (
            BondStereo.STEREONONE,
            BondStereo.STEREOANY,
            BondStereo.STEREOZ,
            BondStereo.STEREOE,
            BondStereo.STEREOCIS,
            BondStereo.STEREOTRANS,
        )

        super().__init__(
            NullityFeaturizer(),
            OneHotFeaturizer(lambda b: b.GetBondType(), self.bond_types),
            ValueFeaturizer(lambda b: b.GetIsConjugated(), int),
            ValueFeaturizer(lambda b: b.IsInRing(), int),
            OneHotFeaturizer(lambda b: b.GetStereo(), self.stereos, padding=True),
        )


class RIGRBondFeaturizer(MultiHotFeaturizer[Bond]):
    """A :class:`RIGRBondFeaturizer` feauturizes bonds based on only the resonance-invariant features:

    * ``null``-ity (i.e., is the bond ``None``?)
    * in ring?

    Example
    -------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C=C-c1ccccc1')
    >>> featurizer = RIGRBondFeaturizer()
    >>> for i in range(4):
    ...     print(featurizer.to_string(mol.GetBondWithIdx(i)))
    0 0
    0 0
    0 1
    0 1

    """

    def __init__(self):
        super().__init__(NullityFeaturizer(), ValueFeaturizer(lambda b: b.IsInRing(), int))
