from typing import List, Sequence

import numpy as np
from rdkit.Chem.rdchem import Bond, BondType

from chemprop.featurizers.base import VectorFeaturizer


class MultiHotBondFeaturizer(VectorFeaturizer[Bond]):
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
    bond_types : Sequence[BondType] | None, default=[SINGLE, DOUBLE, TRIPLE, AROMATIC]
        the known bond types
    stereos : Sequence[int] | None, default=[0, 1, 2, 3, 4, 5]
        the known bond stereochemistries. See [1]_ for more details
    keep_features : List[bool], optional
        a list of booleans to indicate which bond features to keep except for nullity. If None, all features are kept. If False, corresponding feature is set to zeros. Useful for ablation and SHAP analysis.

    References
    ----------
    .. [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondStereo.values
    """

    def __init__(
        self, 
        bond_types: Sequence[BondType] | None = None, 
        stereos: Sequence[int] | None = None,
        keep_features: List[bool] = None,
    ):
        self.bond_types = bond_types or [
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
        ]
        self.stereo = stereos or range(6)

        self.__size = 1 + len(self.bond_types) + 2 + (len(self.stereo) + 1)

        if keep_features is None:
            keep_features = [True] * 4 
        self.keep_features = keep_features        

    def __len__(self) -> int:
        return self.__size

    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)

        if b is None:
            x[0] = 1
            return x

        i = 1
        bond_type = b.GetBondType()
        bt_bit, size = self.one_hot_index(bond_type, self.bond_types)
        if self.keep_features[0] and bt_bit != size:
            x[i + bt_bit] = 1
        i += size - 1

        if self.keep_features[1]:
            x[i] = int(b.GetIsConjugated())
        if self.keep_features[2]:
            x[i + 1] = int(b.IsInRing())
        i += 2

        if self.keep_features[3]:
            stereo_bit, _ = self.one_hot_index(int(b.GetStereo()), self.stereo)
            x[i + stereo_bit] = 1

        return x
    
    def zero_mask(self) -> np.ndarray:
        """featurize the bond by setting all bits to zero"""
        return np.zeros(len(self), int)
        
    @classmethod
    def one_hot_index(cls, x, xs: Sequence) -> tuple[int, int]:
        """Returns a tuple of the index of ``x`` in ``xs`` and ``len(xs) + 1`` if ``x`` is in ``xs``.
        Otherwise, returns a tuple with ``len(xs)`` and ``len(xs) + 1``."""
        n = len(xs)

        return xs.index(x) if x in xs else n, n + 1
