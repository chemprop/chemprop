from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
import os
import pickle
import sys
sys.path.append('../')
from typing import List

import numpy as np
from scipy import sparse
from tqdm import tqdm

from chemprop.data.utils import get_data
from chemprop.features import get_features_func


def save(save_path: str, features: List[List[int]]):
    features = np.stack(features)
    sparse_features = sparse.csr_matrix(features)

    # Write to temporary file first rather than overwriting save_path
    # in case there's a crash during writing
    temp_save_path = save_path + '_temp'

    with open(temp_save_path, 'wb') as f:
        pickle.dump(sparse_features, f)

    os.rename(temp_save_path, save_path)


def save_features(args: Namespace):
    # Get data and features function
    data = get_data(args.data_path, max_data_size=args.max_data_size)
    features_func = get_features_func(args.features_generator, args)

    # Load partially complete data
    features = []
    if os.path.exists(args.save_path):
        with open(args.save_path, 'rb') as f:
            features = pickle.load(f).todense().tolist()

    # Build features map function
    data = data[len(features):]  # restrict to data for which features have not been computed yet
    mols = (d.mol for d in data)
    features_map = map(features_func, mols) if args.sequential else Pool().imap(features_func, mols)
    features_map = tqdm(features_map, total=len(data))

    # Get features
    for i, feats in enumerate(features_map):
        features.append(feats)

        if i > 0 and i % args.save_frequency == 0:
            save(args.save_path, features)

    save(args.save_path, features)


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
