"""Global configuration variables for chemprop"""

from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer

DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM = SimpleMoleculeMolGraphFeaturizer().shape
DEFAULT_HIDDEN_DIM = 300

import lightning
from packaging.version import Version

LIGHTNING_26_COMPAT_ARGS = {}
if Version(lightning.__version__) >= Version("2.6"):
    LIGHTNING_26_COMPAT_ARGS["weights_only"] = False
