"""Global configuration variables for chemprop"""
from chemprop.v2.featurizers.molecule import MoleculeMolGraphFeaturizer


DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM = MoleculeMolGraphFeaturizer().shape
DEFAULT_HIDDEN_DIM = 300
