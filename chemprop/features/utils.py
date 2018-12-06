from argparse import Namespace
from functools import partial
import pickle
from typing import Callable, List, Union

import numpy as np
from rdkit import Chem

from .descriptors import mordred_features
from .morgan_fingerprint import morgan_fingerprint
from .rdkit_features import rdkit_2d_features


def load_features(path: str) -> List[np.ndarray]:
    with open(path, 'rb') as f:
        features = pickle.load(f)
    features = [np.squeeze(np.array(feat.todense())) for feat in features]

    return features


def get_features_func(features_generator: str,
                      args: Namespace = None) -> Union[Callable[[Chem.Mol], np.ndarray],
                                                       partial]:
    if features_generator == 'morgan':
        return partial(morgan_fingerprint, use_counts=False)

    if features_generator == 'morgan_count':
        return partial(morgan_fingerprint, use_counts=True)

    if features_generator == 'rdkit_2d':
        assert args is not None
        assert hasattr(args, 'functional_group_smarts')  # TODO: handle this in a better way
        return partial(rdkit_2d_features, args=args)

    if features_generator == 'mordred':
        return mordred_features

    raise ValueError('features_generator type "{}" not supported.'.format(features_generator))
