from argparse import Namespace
from functools import partial
import os
import pickle
from typing import Callable, List, Union
import logging

import numpy as np
from rdkit import Chem

from .descriptors import mordred_features
from .morgan_fingerprint import morgan_fingerprint
from .rdkit_features import rdkit_2d_features


def load_features(path: str) -> List[np.ndarray]:
    """
    Loads features saved as a .pckl file or as a directory of .pckl files.

    If path is a directory, assumes features are saved in files named 0.pckl, 1.pckl, ...

    :param path: Path to a .pckl file or a directory of .pckl files named as above.
    :return: A list of numpy arrays containing the features.
    """
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            features = pickle.load(f)
        features = [np.squeeze(np.array(feat.todense())) for feat in features]
    else:
        features = []
        features_num = 0
        features_path = os.path.join(path, f'{features_num}.pckl')

        while os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                feats = pickle.load(f)
            features.extend([np.squeeze(np.array(feat.todense())) for feat in feats])

            features_num += 1
            features_path = os.path.join(path, f'{features_num}.pckl')

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

    if features_generator == "rdkit_2d_normalized":
        from .rdkit_normalized_features import rdkit_2d_normalized_features
        if rdkit_2d_normalized_features is None:
            logging.getLogger(__name__).warning("Please install descriptastorus for normalized descriptors")
            raise ValueError(f'feature_generator type "{features_generator}" not installed')
       
        return rdkit_2d_normalized_features

    if features_generator == 'mordred':
        return mordred_features

    raise ValueError(f'features_generator type "{features_generator}" not supported.')
