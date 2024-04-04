import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from chemprop.featurizers.base import VectorFeaturizer
from chemprop.utils import ClassRegistry

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
