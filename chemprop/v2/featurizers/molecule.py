from typing import Protocol

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from chemprop.v2.utils import ClassRegistry

MoleculeFeaturizerRegistry = ClassRegistry()


class MoleculeFeaturizerProto(Protocol):
    """A :class:`MoleculeFeaturizerProto` calculates feature vectors of RDKit molecules."""

    def __len__(self) -> int:
        """the length of the feature vector"""

    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        """Featurize the molecule ``mol``"""


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
        return self.F.GetCountFingerprintAsNumPy(mol)


@MoleculeFeaturizerRegistry("morgan_binary")
class MorganBinaryFeaturzer(MorganFeaturizerMixin, BinaryFeaturizerMixin, MoleculeFeaturizerProto):
    pass


@MoleculeFeaturizerRegistry("morgan_count")
class MorganCountFeaturizer(MorganFeaturizerMixin, CountFeaturizerMixin, MoleculeFeaturizerProto):
    pass


# try:
#     from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

#     # @register_features_generator('rdkit_2d')
#     def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
#         """
#         Generates RDKit 2D features for a molecule.

#         :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
#         :return: A 1D numpy array containing the RDKit 2D features.
#         """
#         smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
#         generator = rdDescriptors.RDKit2D()
#         features = generator.process(smiles)[1:]

#         return features

#     # @register_features_generator('rdkit_2d_normalized')
#     def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
#         """
#         Generates RDKit 2D normalized features for a molecule.

#         :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
#         :return: A 1D numpy array containing the RDKit 2D normalized features.
#         """
#         smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
#         generator = rdNormalizedDescriptors.RDKit2DNormalized()
#         features = generator.process(smiles)[1:]

#         return features
# except ImportError:
#     # @register_features_generator('rdkit_2d')
#     def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
#         """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
#         raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
#                           '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D features.')

#     # @register_features_generator('rdkit_2d_normalized')
#     def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
#         """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
#         raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
#                           '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D normalized features.')


"""
Custom features generator template.

Note: The name you use to register the features generator is the name
you will specify on the command line when using the --features_generator <name> flag.
Ex. python train.py ... --features_generator custom ...

@register_features_generator('custom')
def custom_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    features = np.array([0, 0, 1])

    return features
"""
