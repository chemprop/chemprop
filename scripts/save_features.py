from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
import os
import pickle
import shutil
import sys
sys.path.append('../')
from typing import List, Tuple

import numpy as np
from scipy import sparse
from tqdm import tqdm

from chemprop.data.utils import get_data
from chemprop.features import get_features_func


def load_temp(temp_dir: str) -> Tuple[List[List[float]], int]:
    """
    Loads all features saved as .pckl files in load_dir.

    Assumes temporary files are named in order 0.pckl, 1.pckl, ...

    :param temp_dir: Directory in which temporary .pckl files containing features are stored.
    :return: A tuple with a list of molecule features, where each molecule's features is a list of floats,
    and the number of temporary files.
    """
    features = []
    temp_num = 0
    temp_path = os.path.join(temp_dir, '{}.pckl'.format(temp_num))

    while os.path.exists(temp_path):
        with open(temp_path, 'rb') as f:
            features.extend(pickle.load(f).todense().tolist())

            temp_num += 1
        temp_path = os.path.join(temp_dir, '{}.pckl'.format(temp_num))

    return features, temp_num


def save(save_path: str, features: List[List[int]]):
    """
    Saves features as a sparse 2D array in a .pckl file.

    Assumes temporary files are named in order 0.pckl, 1.pckl, ...

    :param save_path: Path to .pckl file where features will be saved.
    :param features: A list of molecule features, where each molecule's features is a list of floats.
    """
    features = np.stack(features)
    sparse_features = sparse.csr_matrix(features)

    with open(save_path, 'wb') as f:
        pickle.dump(sparse_features, f)


def save_features(args: Namespace):
    """Computes and saves features for a dataset of molecules as a sparse 2D array in a .pckl file."""
    # Get data and features function
    data = get_data(args.data_path, max_data_size=args.max_data_size)
    features_func = get_features_func(args.features_generator, args)
    temp_save_dir = args.save_path + '_temp'

    # Load partially complete data
    if args.restart:
        os.remove(args.save_path)
        shutil.rmtree(temp_save_dir)
    else:
        if os.path.exists(args.save_path):
            raise ValueError('"{}" already exists and args.restart is False.'.format(args.save_path))

        if os.path.exists(temp_save_dir):
            features, temp_num = load_temp(temp_save_dir)

    if not os.path.exists(temp_save_dir):
        os.makedirs(temp_save_dir)
        features, temp_num = [], 0

    # Build features map function
    data = data[len(features):]  # restrict to data for which features have not been computed yet
    mols = (d.mol for d in data)
    if args.sequential:
        features_map = tqdm(map(features_func, mols), total=len(data))
    else:
        with Pool() as pool:
            features_map = tqdm(pool.imap(features_func, mols), total=len(data))

    # Get features
    temp_features = []
    for i, feats in enumerate(features_map):
        temp_features.append(feats)

        # Save temporary features every save_frequency
        if (i > 0 and (i + 1) % args.save_frequency == 0) or i == len(data) - 1:
            save(os.path.join(temp_save_dir, '{}.pckl'.format(temp_num)), temp_features)
            features.extend(temp_features)
            temp_features = []
            temp_num += 1

    try:
        # Save all features
        save(args.save_path, features)

        # Remove temporary features
        shutil.rmtree(temp_save_dir)
    except OverflowError:
        print('Features array is too large to save as a single file. Instead keeping features as a directory of files.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV')
    parser.add_argument('--features_generator', type=str, required=True,
                        choices=['morgan', 'morgan_count', 'rdkit_2d', 'mordred'],
                        help='Type of features to generate')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to .pckl file where features will be saved as a Python pickle file')
    parser.add_argument('--save_frequency', type=int, default=10000,
                        help='Frequency with which to save the features')
    parser.add_argument('--restart', action='store_true', default=False,
                        help='Whether to not load partially complete featurization and instead start from scratch')
    parser.add_argument('--functional_group_smarts', type=str, default='../chemprop/features/smarts.txt',
                        help='Path to txt file of smarts for functional groups, if functional_group features are on.')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Whether to run sequential rather than in parallel')
    args = parser.parse_args()

    dirname = os.path.dirname(args.save_path)
    if dirname != '':
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    save_features(args)
