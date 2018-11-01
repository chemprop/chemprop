import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def morgan_fingerprint(smiles: str, radius: int = 3, num_bits: int = 2048, use_counts: bool = False) -> np.ndarray:
    """
    Generates a morgan fingerprint for a smiles string.

    :param smiles: A smiles string for a molecule.
    :param radius: The radius of the fingerprint.
    :param num_bits: The number of bits to use in the fingerprint.
    :param use_counts: Whether to use counts or just a bit vector for the fingerprint
    :return: A 1-D numpy array containing the morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(smiles)
    if use_counts:
        fp_vect = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    else:
        fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vect, fp)

    return fp
