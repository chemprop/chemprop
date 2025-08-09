import logging

from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
import multiprocess
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from chemprop.featurizers.base import VectorFeaturizer
from chemprop.utils import ClassRegistry

logger = logging.getLogger(__name__)

MoleculeFeaturizerRegistry = ClassRegistry[VectorFeaturizer[Mol]]()


class MorganFeaturizerMixin:
    def __init__(self, radius: int = 2, length: int = 2048, include_chirality: bool = True):
        if radius < 0:
            raise ValueError(f"arg 'radius' must be >= 0! got: {radius}")

        self.length = length
        self.F = GetMorganGenerator(
            radius=radius, fpSize=length, includeChirality=include_chirality
        )

    def __len__(self) -> int:
        return self.length


class BinaryFeaturizerMixin:
    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        return self.F.GetFingerprintAsNumPy(mol)


class CountFeaturizerMixin:
    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        return self.F.GetCountFingerprintAsNumPy(mol).astype(np.int32)


@MoleculeFeaturizerRegistry("morgan_binary")
class MorganBinaryFeaturizer(MorganFeaturizerMixin, BinaryFeaturizerMixin, VectorFeaturizer[Mol]):
    pass


@MoleculeFeaturizerRegistry("morgan_count")
class MorganCountFeaturizer(MorganFeaturizerMixin, CountFeaturizerMixin, VectorFeaturizer[Mol]):
    pass


@MoleculeFeaturizerRegistry("rdkit_2d")
class RDKit2DFeaturizer(VectorFeaturizer[Mol]):
    def __init__(self):
        if multiprocess.current_process().name == "MainProcess":
            logger.warning(
                "The RDKit 2D features can deviate signifcantly from a normal distribution. Consider "
                "manually scaling them using an appropriate scaler before creating datapoints, rather "
                "than using the scikit-learn `StandardScaler` (the default in Chemprop)."
            )

    def __len__(self) -> int:
        return len(Descriptors.descList)

    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        features = np.array(
            [
                0.0 if name == "SPS" and mol.GetNumHeavyAtoms() == 0 else func(mol)
                for name, func in Descriptors.descList
            ],
            dtype=float,
        )

        return features


class V1RDKit2DFeaturizerMixin(VectorFeaturizer[Mol]):
    def __len__(self) -> int:
        return 200

    def __call__(self, mol: Mol) -> np.ndarray:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        features = self.generator.process(smiles)[1:]

        return np.array(features)


@MoleculeFeaturizerRegistry("v1_rdkit_2d")
class V1RDKit2DFeaturizer(V1RDKit2DFeaturizerMixin):
    def __init__(self):
        self.generator = rdDescriptors.RDKit2D()


@MoleculeFeaturizerRegistry("v1_rdkit_2d_normalized")
class V1RDKit2DNormalizedFeaturizer(V1RDKit2DFeaturizerMixin):
    def __init__(self):
        self.generator = rdNormalizedDescriptors.RDKit2DNormalized()


@MoleculeFeaturizerRegistry("charge")
class ChargeFeaturizer(VectorFeaturizer[Mol]):
    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        return np.array([Chem.GetFormalCharge(mol)])

    def __len__(self) -> int:
        return 1
