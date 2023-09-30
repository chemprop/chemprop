"""Global configuration variables for chemprop"""
from chemprop.v2.featurizers.molgraph import MolGraphFeaturizer


DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM = MolGraphFeaturizer().shape
DEFAULT_HIDDEN_DIM = 300
