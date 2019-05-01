from typing import Callable, List, Union

import numpy as np
import pandas as pd
try:
    from openeye.oechem import *
    RDKIT=False
    Molecule = Union[str, OEGraphMol]
except:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    RDKIT=True
    Molecule = Union[str, Chem.Mol]

FeaturesGenerator = Callable[[Molecule], np.ndarray]
#Descs_DF = pd.read_pickle("/tmp/rrck_rdsmiles.pkl")

FEATURES_GENERATOR_REGISTRY = {}


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


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """
    Gets a registered FeaturesGenerator by name.

    :param features_generator_name: The name of the FeaturesGenerator.
    :return: The desired FeaturesGenerator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns the names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())


if RDKIT:
    MORGAN_RADIUS = 2
    MORGAN_NUM_BITS = 2048

    @register_features_generator('morgan')
    def morgan_binary_features_generator(mol: Molecule,
                                         radius: int = MORGAN_RADIUS,
                                         num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
        """
        Generates a binary Morgan fingerprint for a molecule.

        :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
        :param radius: Morgan fingerprint radius.
        :param num_bits: Number of bits in Morgan fingerprint.
        :return: A 1-D numpy array containing the binary Morgan fingerprint.
        """
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)

        return features


    @register_features_generator('morgan_count')
    def morgan_counts_features_generator(mol: Molecule,
                                         radius: int = MORGAN_RADIUS,
                                         num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
        """
        Generates a counts-based Morgan fingerprint for a molecule.

        :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
        :param radius: Morgan fingerprint radius.
        :param num_bits: Number of bits in Morgan fingerprint.
        :return: A 1D numpy array containing the counts-based Morgan fingerprint.
        """
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
        features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)

        return features

try:
    import rdkit
    @register_features_generator('file_cache')
    def cached_dragon_features(mol: Molecule,) -> np.ndarray:
        """
        Pulls descriptors from an external file by a smiles join
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        desc = Descs_DF.loc[[smiles]].values
        return desc[0]    
except ImportError:
    pass

try:
    from openeye.oechem import *
    @register_features_generator('file_cache')
    def cached_dragon_features(mol: Molecule,) -> np.ndarray:
        """
        Pulls descriptors from an external file by a smiles join
        """
        smiles = OECreateSmiString(mol, OESMILESFlag_ISOMERIC) 
        desc = Descs_DF.loc[[smiles]].values
        return desc[0]
except ImportError:
    pass


try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D features for a molecule.

        :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]

        return features

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule.

        :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]

        return features
except ImportError:
    pass


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
