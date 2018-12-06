from mordred import Calculator, descriptors
import numpy as np
from rdkit import Chem


mordred_calc = Calculator(descriptors, ignore_3D=True)  # can't do 3D without sdf or mol file


def mordred_features(mol: Chem.Mol) -> np.ndarray:
    return np.array([float(f) for f in mordred_calc(mol)])
