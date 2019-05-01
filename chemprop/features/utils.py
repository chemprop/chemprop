import csv
import os
import pickle
from typing import List

import numpy as np
import pandas as pd

def save_features(path: str, features: List[np.ndarray]):
    """
    Saves features to a compressed .npz file with array name "features".

    :param path: Path to a .npz file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")
    - .npz (assumes features are saved with name "features")
    - .npy
    - .csv/.txt (assumes comma-separated features with a header and with one line per molecule)
    - .pkl/.pckl/.pickle containing a sparse numpy array (TODO: remove this option once we are no longer dependent on it)

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]
    print("Trying to load features from %s" % extension)
    #I've broken all other forms because id-less features is asking for pain
    if extension == '.npz':
        features = np.load(path)['features']
        raise ValueError('Braindead feature loading ignored')
    elif extension == '.npy':
        features = np.load(path)
        raise ValueError('Braindead feature loading ignored')
    elif extension in ['.csv', '.txt','.test']:
        # NAW: Blech.  our input will be csv so I'm just fixing this.  We assume a id,desc,desc,desc
        df = pd.read_csv(path, index_col=0, header=0)
        # Rather than do the indexing later, I'll just create a lookup to numpy array
        features = {}
        for idx, row in df.iterrows():
            features[idx] = row.values
    elif extension in ['.pkl', '.pckl', '.pickle']:
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
            raise ValueError('Braindead feature loading ignored')
    else:
        raise ValueError(f'Features path extension {extension} not supported.')
    return features
