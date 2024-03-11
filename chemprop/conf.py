"""Global configuration variables for chemprop"""

from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer, MultiHotAtomFeaturizer


DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.default()).shape
DEFAULT_HIDDEN_DIM = 300
