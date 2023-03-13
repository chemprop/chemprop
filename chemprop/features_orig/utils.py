import csv
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools


def save_features(path: str, features: List[np.ndarray]) -> None:
    """
    Saves features to a compressed :code:`.npz` file with array name "features".

    :param path: Path to a :code:`.npz` file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:

    * :code:`.npz` compressed (assumes features are saved with name "features")
    * .npy
    * :code:`.csv` / :code:`.txt` (assumes comma-separated features with a header and with one line per molecule)
    * :code:`.pkl` / :code:`.pckl` / :code:`.pickle` containing a sparse numpy array

    .. note::

       All formats assume that the SMILES loaded elsewhere in the code are in the same
       order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size :code:`(num_molecules, features_size)` containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    elif extension == '.npy':
        features = np.load(path)
    elif extension in ['.csv', '.txt']:
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            features = np.array([[float(value) for value in row] for row in reader])
    elif extension in ['.pkl', '.pckl', '.pickle']:
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


def load_valid_atom_or_bond_features(path: str, smiles: List[str]) -> List[np.ndarray]:
    """
    Loads features saved in a variety of formats.

    Supported formats:

    * :code:`.npz` descriptors are saved as 2D array for each molecule in the order of that in the data.csv
    * :code:`.pkl` / :code:`.pckl` / :code:`.pickle` containing a pandas dataframe with smiles as index and numpy array of descriptors as columns
    * :code:'.sdf' containing all mol blocks with descriptors as entries

    :param path: Path to file containing atomwise features.
    :return: A list of 2D array.
    """

    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        container = np.load(path)
        features = [container[key] for key in container]

    elif extension in ['.pkl', '.pckl', '.pickle']:
        features_df = pd.read_pickle(path)
        if features_df.iloc[0, 0].ndim == 1:
            features = features_df.apply(lambda x: np.stack(x.tolist(), axis=1), axis=1).tolist()
        elif features_df.iloc[0, 0].ndim == 2:
            features = features_df.apply(lambda x: np.concatenate(x.tolist(), axis=1), axis=1).tolist()
        else:
            raise ValueError(f'Atom/bond descriptors input {path} format not supported')

    elif extension == '.sdf':
        features_df = PandasTools.LoadSDF(path).drop(['ID', 'ROMol'], axis=1).set_index('SMILES')

        features_df = features_df[~features_df.index.duplicated()]

        # locate atomic descriptors columns
        features_df = features_df.iloc[:, features_df.iloc[0, :].apply(lambda x: isinstance(x, str) and ',' in x).to_list()]
        features_df = features_df.reindex(smiles)
        if features_df.isnull().any().any():
            raise ValueError('Invalid custom atomic descriptors file, Nan found in data')

        features_df = features_df.applymap(lambda x: np.array(x.replace('\r', '').replace('\n', '').split(',')).astype(float))

        features = features_df.apply(lambda x: np.stack(x.tolist(), axis=1), axis=1).tolist()

    else:
        raise ValueError(f'Extension "{extension}" is not supported.')

    return features
