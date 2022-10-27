from typing import Mapping, Optional
from .modules import (
    MessagePassingBlock,
    MolecularMessagePassingBlock,
    MolecularInput,
    CompositeMessagePassingBlock,
    AtomMessageBlock,
    BondMessageBlock,
)
from .models import (
    MPNN,
    ClassificationMPNN,
    BinaryClassificationMPNN,
    DirichletClassificationMPNN,
    MulticlassMPNN,
    DirichletMulticlassMPNN,
    RegressionMPNN,
    MveRegressionMPNN,
    EvidentialMPNN,
    SpectralMPNN,
)
from .loss import LossFunction, build_loss
