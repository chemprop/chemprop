from argparse import Namespace
import pickle
from typing import Callable, List, Optional

from mordred import Calculator, descriptors
import numpy as np
from rdkit import Chem

from .morgan_fingerprint import morgan_fingerprint
from .rdkit_features import rdkit_2d_features


mordred_calc = Calculator(descriptors, ignore_3D=True)  # can't do 3D without sdf or mol file


def load_features(path: str) -> List[np.ndarray]:
    with open(path, 'rb') as f:
        features = pickle.load(f)
    features = [np.squeeze(np.array(feat.todense())) for feat in features]

    return features


def get_features_func(features_generator: str) -> Callable[[Chem.Mol, Optional[Namespace]], np.ndarray]:
    if features_generator == 'morgan':
        def features_func(mol: Chem.Mol, args: Namespace = None):
            return morgan_fingerprint(mol, use_counts=False)
    elif features_generator == 'morgan_count':
        def features_func(mol: Chem.Mol, args: Namespace = None):
            return morgan_fingerprint(mol, use_counts=True)
    elif features_generator == 'rdkit_2d':
        def features_func(mol: Chem.Mol, args: Namespace = None):
            return rdkit_2d_features(mol, args)
    elif features_generator == 'mordred':
        def features_func(mol: Chem.Mol, args: Namespace = None):
            return [float(f) for f in mordred_calc(mol)]
    else:
        raise ValueError('features_generator type "{}" not supported.'.format(features_generator))

    return features_func
