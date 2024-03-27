"""Global configuration variables for chemprop"""

from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer


DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM = SimpleMoleculeMolGraphFeaturizer().shape
DEFAULT_HIDDEN_DIM = 300
