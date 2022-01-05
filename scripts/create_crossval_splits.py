from copy import deepcopy
import os
import pickle
import random
from typing import List
from typing_extensions import Literal


import numpy as np
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from chemprop.data import MoleculeDataset
from chemprop.data import get_data, scaffold_to_smiles


class Args(Tap):
    data_path: str  # Path to CSV file with dataset of molecules
    save_dir: str  # Path to CSV file where splits will be saved
    split_type: Literal['random', 'scaffold', 'time_window']  # Random or scaffold based split
    num_folds: int = 10  # Number of cross validation folds
    test_folds_to_test: int = 3  # Number of test folds
    val_folds_per_test: int = 3  # Number of val folds
    time_folds_per_train_set: int = 3  # X:1:1 train:val:test for time split sliding window
    smiles_columns: List[str] = None # columns in CSV dataset file containing SMILES
    split_key_molecule: int = 0 # index of the molecule to use for splitting in muli-molecule data


def split_indices(all_indices: List[int],
                  num_folds: int,
                  scaffold: bool = False,
                  split_key_molecule: int = 0,
                  data: MoleculeDataset = None,
                  shuffle: bool = True) -> List[List[int]]:
    num_data = len(all_indices)
    if scaffold:
        key_mols = [m[split_key_molecule] for m in data.mols(flatten=False)]
        scaffold_to_indices = scaffold_to_smiles(key_mols, use_indices=True)
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)
        fold_indices = [[] for _ in range(num_folds)]
        for s in index_sets:
            length_array = [len(fi) for fi in fold_indices]
            min_index = length_array.index(min(length_array))
            fold_indices[min_index] += s
        if shuffle:
            random.shuffle(fold_indices)
    else:  # random
        if shuffle:
            random.shuffle(all_indices)
        fold_indices = []
        for i in range(num_folds):
            begin, end = int(i * num_data / num_folds), int((i + 1) * num_data / num_folds)
            fold_indices.append(np.array(all_indices[begin:end]))
    return fold_indices


def create_time_splits(args: Args):
    # ASSUME DATA GIVEN IN CHRONOLOGICAL ORDER.
    # this will dump a very different format of indices, with all in one file; TODO modify as convenient later.
    data = get_data(path=args.data_path, smiles_columns=args.smiles_columns)
    num_data = len(data)
    all_indices = list(range(num_data))
    fold_indices = {'random': [], 'scaffold': [], 'time': []}
    for i in range(args.num_folds - args.time_folds_per_train_set - 1):
        begin, end = int(i * num_data / args.num_folds), int(
            (i + args.time_folds_per_train_set + 2) * num_data / args.num_folds)
        subset_indices = all_indices[begin:end]
        subset_data = MoleculeDataset(data[begin:end])
        fold_indices['random'].append(split_indices(deepcopy(subset_indices), args.time_folds_per_train_set + 2))
        fold_indices['scaffold'].append(
            split_indices(subset_indices, args.time_folds_per_train_set + 2, scaffold=True, split_key_molecule=args.split_key_molecule ,data=subset_data))
        fold_indices['time'].append(split_indices(subset_indices, args.time_folds_per_train_set + 2, shuffle=False))
    for split_type in ['random', 'scaffold', 'time']:
        all_splits = []
        for i in range(len(fold_indices[split_type])):
            os.makedirs(os.path.join(args.save_dir, split_type, 'fold_' + str(i), '0'), exist_ok=True)
            with open(os.path.join(args.save_dir, split_type, 'fold_' + str(i), '0', 'split_indices.pckl'), 'wb') as wf:
                train = np.concatenate([fold_indices[split_type][i][j] for j in range(args.time_folds_per_train_set)])
                # train = []
                # for fold in train_folds:
                #     train += fold
                val = fold_indices[split_type][i][-2]
                test = fold_indices[split_type][i][-1]
                pickle.dump([train, val, test],
                            wf)  # each is a pickle file containing a list of length-3 index lists for train/val/test
                all_splits.append([train, val, test])
        with open(os.path.join(args.save_dir, split_type, 'fold_' + str(i), 'split_indices.pckl'), 'wb') as wf:
            pickle.dump(all_splits, wf)


def create_crossval_splits(args: Args):
    data = get_data(path=args.data_path, smiles_columns=args.smiles_columns)
    num_data = len(data)
    if args.split_type == 'random':
        all_indices = list(range(num_data))
        fold_indices = split_indices(all_indices, args.num_folds, scaffold=False)
    elif args.split_type == 'scaffold':
        all_indices = list(range(num_data))
        fold_indices = split_indices(all_indices, args.num_folds, scaffold=True, split_key_molecule=args.split_key_molecule, data=data)
    else:
        raise ValueError
    random.shuffle(fold_indices)
    for i in range(args.test_folds_to_test):
        all_splits = []
        for j in range(1, args.val_folds_per_test + 1):
            os.makedirs(os.path.join(args.save_dir, args.split_type, f'fold_{i}', f'{j - 1}'), exist_ok=True)
            with open(os.path.join(args.save_dir, args.split_type, f'fold_{i}', f'{j - 1}', 'split_indices.pckl'),
                      'wb') as wf:
                val_idx = (i + j) % args.num_folds
                val = fold_indices[val_idx]
                test = fold_indices[i]
                train = []
                for k in range(args.num_folds):
                    if k != i and k != val_idx:
                        train.append(fold_indices[k])
                train = np.concatenate(train)
                pickle.dump([train, val, test], wf)
                all_splits.append([train, val, test])
        with open(os.path.join(args.save_dir, args.split_type, f'fold_{i}', 'split_indices.pckl'), 'wb') as wf:
            pickle.dump(all_splits, wf)


if __name__ == '__main__':
    args = Args().parse_args()

    random.seed(0)

    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.data_path)
        if args.split_type == 'time_window':
            args.save_dir = os.path.join(args.save_dir, 'time_window')

    if args.split_type == 'time_window':
        create_time_splits(args)
    else:
        create_crossval_splits(args)
