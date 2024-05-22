import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol, Descriptors
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


@MoleculeFeaturizerRegistry("rdkit_2d")
class RDKit2DFeaturizer(VectorFeaturizer[Mol]):
    def __init__(self):
        pass

    def __len__(self) -> int:
        return len(Descriptors.descList)

    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        if mol.GetNumHeavyAtoms() == 0:
            features = np.array(
                [0 if name == "SPS" else func(mol) for name, func in sorted(Descriptors.descList)]
            )
        else:
            features = np.array([func(mol) for name, func in sorted(Descriptors.descList)])

        NAN_TOKEN = 0
        features[np.isnan(self.x_d)] = NAN_TOKEN
        return features
