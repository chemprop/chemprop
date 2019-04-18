from argparse import ArgumentParser
import pickle
import os
import random

import numpy as np

from chemprop.data.utils import get_data


def create_crossval_splits(args):
    data = get_data(args.data_path)
    num_data = len(data)
    if args.split_type == 'random':
        all_indices = list(range(num_data))
        random.shuffle(all_indices)
        fold_indices = []
        for i in range(args.num_folds):
            begin, end = int(i * num_data / args.num_folds), int((i+1) * num_data / args.num_folds)
            fold_indices.append(np.array(all_indices[begin:end]))
    elif args.split_type == 'scaffold':
        raise NotImplementedError  # TODO
    elif args.split_type == 'time': # same as random, but without shuffling. ASSUME DATA GIVEN IN CHRONOLOGICAL ORDER. 
        all_indices = list(range(num_data))
        fold_indices = []
        for i in range(args.num_folds):
            begin, end = int(i * num_data / args.num_folds), int((i+1) * num_data / args.num_folds)
            fold_indices.append(np.array(all_indices[begin:end]))
    else:
        raise ValueError
    os.makedirs(os.path.join(args.save_dir, args.split_type), exist_ok=True)
    for i in range(args.num_folds):
        with open(os.path.join(args.save_dir, args.split_type, f'{i}.pkl'), 'wb') as wf:
            pickle.dump(fold_indices[i], wf)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file with dataset of molecules')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to CSV file where splits will be saved')
    parser.add_argument('--split_type', type=str, choices=['random', 'scaffold', 'time'], required=True,
                        help='Random or scaffold based split')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='Number of cross validation folds')
    args = parser.parse_args()

    create_crossval_splits(args)
