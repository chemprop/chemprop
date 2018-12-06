from argparse import ArgumentParser, Namespace
import os
import pickle
import sys
sys.path.append('../')

import numpy as np
from scipy import sparse
from tqdm import tqdm

from chemprop.data.utils import get_data
from chemprop.features import get_features_func


def save_features(args: Namespace):
    data = get_data(args.data_path, max_data_size=args.max_data_size)
    features_func = get_features_func(args.features_generator)
    features = np.stack([features_func(d.mol, args) for d in tqdm(data, total=len(data))])
    sparse_features = sparse.csr_matrix(features)

    with open(args.save_path, 'wb') as f:
        pickle.dump(sparse_features, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV')
    parser.add_argument('--features_generator', type=str, required=True,
                        choices=['morgan', 'morgan_count', 'rdkit_2d', 'mordred'],
                        help='Type of features to generate')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to .pckl file where features will be saved as a Python pickle file')
    parser.add_argument('--functional_group_smarts', type=str, default='../chemprop/features/smarts.txt',
                        help='Path to txt file of smarts for functional groups, if functional_group features are on.')
    parser.add_argument('--max_data_size', type=int, default=None,
                        help='Maximum number of data points to load')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path))

    save_features(args)
