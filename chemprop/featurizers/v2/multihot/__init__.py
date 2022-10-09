from .atom import AtomFeaturizer
from .bond import BondFeaturizer
from .base import MultiHotFeaturizer

_DEFAULT_ATOM_FDIM = len(AtomFeaturizer())
_DEFAULT_BOND_FDIM = len(BondFeaturizer())
