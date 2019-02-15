from abc import ABC, abstractmethod
from typing import Callable, List, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


FEATURES_GENERATOR_REGISTRY = {}


class FeaturesGenerator(ABC):
    """An abstract class for a classes which generate features for a molecule."""

    @abstractmethod
    def __call__(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates features for a molecule.

        :param mol: Either a SMILES string or an RDKit molecule.
        :return: A 1-D numpy array containing the features for the molecule.
        """
        pass


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    Registers a features generator.

    :param features_generator_name: The name to call the FeaturesGenerator.
    :return: A decorator which will add a FeaturesGenerator to the registry using the specified name.
    """
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_features_generator(features_generator_name: str, **kwargs) -> FeaturesGenerator:
    """
    Gets a registered FeaturesGenerator by name.

    :param features_generator_name: The name of the FeaturesGenerator.
    :param kwargs: Keyword arguments for the FeaturesGenerator.
    :return: The desired FeaturesGenerator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_GENERATOR_REGISTRY[features_generator_name](**kwargs)


def get_available_features_generators() -> List[str]:
    """Returns the names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())


class MorganBaseFeaturesGenerator(FeaturesGenerator, ABC):
    """A Morgan Fingerprint generator which can produce either binary or count-based Morgan fingerprints."""

    def __init__(self, radius: int = 2, num_bits: int = 2048):
        """
        Initializes the MorganFeaturesGenerator.

        :param radius: The radius of the fingerprint.
        :param num_bits: The number of bits to use in the fingerprint.
        """
        super(MorganBaseFeaturesGenerator, self).__init__()

        self.radius = radius
        self.num_bits = num_bits

    @property
    @abstractmethod
    def use_counts(self) -> bool:
        """Whether to use counts instead of bits."""
        pass

    def __call__(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates features for a molecule.

        :param mol: Either a SMILES string or an RDKit molecule.
        :return: A 1-D numpy array containing the features for the molecule.
        """
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
        if self.use_counts:
            features_vec = AllChem.GetHashedMorganFingerprint(mol, self.radius, nBits=self.num_bits)
        else:
            features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.num_bits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)

        return features


@register_features_generator('morgan')
class MorganBinaryFeaturesGenerator(MorganBaseFeaturesGenerator):
    @property
    def use_counts(self) -> bool:
        """Whether to use counts instead of bits."""
        return False


@register_features_generator('morgan_count')
class MorganBinaryFeaturesGenerator(MorganBaseFeaturesGenerator):
    @property
    def use_counts(self) -> bool:
        """Whether to use counts instead of bits."""
        return True


try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

    class RDKit2DBaseFeaturesGenerator(FeaturesGenerator, ABC):
        """Abstract base class for RDKit FeaturesGenerators."""

        @property
        @abstractmethod
        def generator(self) -> rdDescriptors.DescriptorGenerator:
            """Returns an RDKit descriptor generator."""
            pass

        def __call__(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
            """
            Generates features for a molecule.

            :param mol: Either a SMILES string or an RDKit molecule.
            :return: A 1-D numpy array containing the features for the molecule.
            """
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
            features = self.generator.process(smiles)[1:]

            return features

    @register_features_generator('rdkit_2d')
    class RDKit2DFeaturesGenerator(RDKit2DBaseFeaturesGenerator):
        @property
        def generator(self) -> rdDescriptors.DescriptorGenerator:
            return rdDescriptors.RDKit2D()

    @register_features_generator('rdkit_2d_normalized')
    class RDKit2DNormalizedFeaturesGenerator(RDKit2DBaseFeaturesGenerator):
        @property
        def generator(self) -> rdDescriptors.DescriptorGenerator:
            return rdNormalizedDescriptors.RDKit2DNormalized()
except ImportError:
    pass

