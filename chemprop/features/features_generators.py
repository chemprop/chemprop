from typing import Callable, List, Union, Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]


FEATURES_GENERATOR_REGISTRY = {}


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    Creates a decorator which registers a features generator in a global dictionary to enable access by name.

    :param features_generator_name: The name to use to access the features generator.
    :return: A decorator which will add a features generator to the registry using the specified name.
    """
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """
    Gets a registered features generator by name.

    :param features_generator_name: The name of the features generator.
    :return: The desired features generator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())


MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


@register_features_generator('morgan')
def morgan_binary_features_generator(mol_data: Union[Molecule, List[List[Molecule]]],
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.
    :param mol_data: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the binary Morgan fingerprint.
    """

    def fingerprint_single_molecule(m: Molecule) -> np.ndarray:
        m = Chem.MolFromSmiles(m) if isinstance(m, str) else m
        features_vec = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=num_bits)
        single_mol_features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, single_mol_features)
        return single_mol_features

    if isinstance(mol_data, Iterable) and not isinstance(mol_data, str):
        features = np.array([[fingerprint_single_molecule(mol) for mol in datum] for datum in mol_data])
    else:
        features = fingerprint_single_molecule(mol_data)

    return features


@register_features_generator('morgan_count')
def morgan_counts_features_generator(mol_data: Union[Molecule, List[List[Molecule]]],
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a counts-based Morgan fingerprint for a molecule.

    :param mol_data: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """

    def fingerprint_single_molecule(m: Molecule) -> np.ndarray:
        m = Chem.MolFromSmiles(m) if isinstance(m, str) else m
        features_vec = AllChem.GetHashedMorganFingerprint(m, radius, nBits=num_bits)
        single_mol_features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, single_mol_features)
        return single_mol_features

    if isinstance(mol_data, Iterable) and not isinstance(mol_data, str):
        features = np.array([[fingerprint_single_molecule(mol) for mol in datum] for datum in mol_data])
    else:
        features = fingerprint_single_molecule(mol_data)

    return features


try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors


    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol_data: Union[Molecule, List[List[Molecule]]]) -> np.ndarray:
        """
        Generates RDKit 2D features for a molecule.
        :param mol_data: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features.
        """
        generator = rdDescriptors.RDKit2D()

        def fingerprint_single_molecule(m: Molecule) -> np.ndarray:
            smiles = Chem.MolToSmiles(m, isomericSmiles=True) if type(m) != str else m
            single_mol_features = np.array(generator.process(smiles)[1:])
            return single_mol_features

        if isinstance(mol_data, Iterable) and not isinstance(mol_data, str):
            features = np.array([[fingerprint_single_molecule(mol) for mol in datum] for datum in mol_data])
        else:
            features = fingerprint_single_molecule(mol_data)

        return features

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol_data: Union[Molecule, List[List[Molecule]]]) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule.

        :param mol_data: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features.
        """
        generator = rdNormalizedDescriptors.RDKit2DNormalized()

        def fingerprint_single_molecule(m: Molecule) -> np.ndarray:
            smiles = Chem.MolToSmiles(m, isomericSmiles=True) if type(m) != str else m
            single_mol_features = np.array(generator.process(smiles)[1:])
            return single_mol_features

        if isinstance(mol_data, Iterable) and not isinstance(mol_data, str):
            features = np.array([[fingerprint_single_molecule(mol) for mol in datum] for datum in mol_data])
        else:
            features = fingerprint_single_molecule(mol_data)

        return features
except ImportError:
    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol_data: Union[Molecule, List[List[Molecule]]]) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                          '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D features.')

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol_data: Union[Molecule, List[List[Molecule]]]) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                          '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D normalized features.')


"""
Custom features generator template.

Note: The name you use to register the features generator is the name
you will specify on the command line when using the --features_generator <name> flag.
Ex. python train.py ... --features_generator custom ...

@register_features_generator('custom')
def custom_features_generator(mol_data: Union[Molecule, List[List[Molecule]]]) -> np.ndarray:
    if isinstance(mol_data, Iterable) and not isinstance(mol_data, AnyStr):
        # If your generator supports an input of a list of molecules, implement  
    
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    features = np.array([0, 0, 1])

    return features
"""
