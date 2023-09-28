from collections import defaultdict
from typing import Sequence

import numpy as np


from chemprop.v2.data.datapoints import MoleculeDatapoint
from chemprop.v2.data.datasets import MoleculeDataset

from astartes import train_val_test_split, train_test_split
from astartes.molecules import train_val_test_split_molecules, train_test_split_molecules

import copy
from logging import Logger
import pickle
from random import Random
from typing import Tuple
import os

from chemprop.data import log_scaffold_stats
from chemprop.args import TrainArgs


def split_data(
    data: Sequence[MoleculeDatapoint],
    split: str = "random",
    sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    key_molecule_index: int = 0,
    seed: int = 0,
    num_folds: int = 1,
    args: TrainArgs = None,
    logger: Logger = None,
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    r"""
    Splits data into training, validation, and test splits.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param key_molecule_index: For data with multiple molecules, this sets which molecule will be considered during splitting.
    :param seed: The random seed to use before shuffling data.
    :param num_folds: Number of folds to create (only needed for "cv" split type).
    :param args: A :class:`~chemprop.args.TrainArgs` object.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Invalid train/val/test splits! got: {sizes}")

    random = Random(seed)

    if args is not None:
        folds_file, val_fold_index, test_fold_index = (
            args.folds_file,
            args.val_fold_index,
            args.test_fold_index,
        )
    else:
        folds_file = val_fold_index = test_fold_index = None

    # typically include a validation set
    include_val = True
    split_fun = train_val_test_split
    mol_split_fun = train_val_test_split_molecules
    # default sampling arguments for astartes sampler
    astartes_kwargs = dict(
        train_size=sizes[0], test_size=sizes[2], return_indices=True, random_state=seed
    )
    # if no validation set, reassign the splitting functions
    if not sizes[1]:
        include_val = False
        split_fun = train_test_split
        mol_split_fun = train_test_split_molecules
    else:
        astartes_kwargs["val_size"] = sizes[1]

    train, val, test = None, None, None
    if split == "crossval":
        index_set = args.crossval_index_sets[args.seed]
        data_split = []
        for split in range(3):
            split_indices = []
            for index in index_set[split]:
                with open(os.path.join(args.crossval_index_dir, f"{index}.pkl"), "rb") as rf:
                    split_indices.extend(pickle.load(rf))
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)

    elif split in {"cv", "cv-no-test"}:
        if num_folds <= 1 or num_folds > len(data):
            raise ValueError(
                "Number of folds for cross-validation must be between 2 and len(data), inclusive."
            )

        random = Random(0)

        indices = np.repeat(np.arange(num_folds), 1 + len(data) // num_folds)[: len(data)]
        random.shuffle(indices)
        test_index = seed % num_folds
        val_index = (seed + 1) % num_folds

        train, val, test = [], [], []
        for d, index in zip(data, indices):
            if index == test_index and split_type != "cv-no-test":
                test.append(d)
            elif index == val_index:
                val.append(d)
            else:
                train.append(d)

    elif split == "index_predetermined":
        split_indices = args.crossval_index_sets[args.seed]

        if len(split_indices) != 3:
            raise ValueError("Split indices must have three splits: train, validation, and test")

        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)

    elif split == "predetermined":
        if not val_fold_index and sizes[2] != 0:
            raise ValueError(
                "Test size must be zero since test set is created separately "
                "and we want to put all other data in train and validation"
            )

        if folds_file is None:
            raise ValueError('arg "folds_file" can not be None!')
        if test_fold_index is None:
            raise ValueError('arg "test_fold_index" can not be None!')

        try:
            with open(folds_file, "rb") as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, "rb") as f:
                all_fold_indices = pickle.load(
                    f, encoding="latin1"
                )  # in case we're loading indices from python2

        log_scaffold_stats(data, all_fold_indices, logger=logger)

        folds = [[data[i] for i in fold_indices] for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != test_fold_index and (val_fold_index is None or i != val_fold_index):
                train_val.extend(folds[i])

        if val_fold_index is not None:
            train = train_val
        else:
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

    elif split == "scaffold_balanced":
        mols_without_atommaps = []
        for mol in data.mols(flatten=False):
            copied_mol = copy.deepcopy(mol[key_molecule_index])
            for atom in copied_mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mols_without_atommaps.append([copied_mol])
        result = mol_split_fun(
            np.array(mols_without_atommaps), sampler="scaffold", **astartes_kwargs
        )
        train, val, test = _unpack_astartes_result(data, result, include_val, log_stats=True)

    elif (
        split == "random_with_repeated_smiles"
    ):  # Use to constrain data with the same smiles go in the same split.
        smiles_dict = defaultdict(set)
        for i, smiles in enumerate(data.smiles()):
            smiles_dict[smiles[key_molecule_index]].add(i)
        index_sets = list(smiles_dict.values())
        random.seed(seed)
        random.shuffle(index_sets)
        train, val, test = [], [], []
        train_size = int(sizes[0] * len(data))
        val_size = int(sizes[1] * len(data))
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
            elif len(val) + len(index_set) <= val_size:
                val += index_set
            else:
                test += index_set
        train = [data[i] for i in train]
        val = [data[i] for i in val]
        test = [data[i] for i in test]

    elif split == "random":
        result = split_fun(np.arange(len(data)), sampler="random", **astartes_kwargs)
        train, val, test = _unpack_astartes_result(data, result, include_val)

    elif split == "kmeans":
        result = mol_split_fun(
            np.array([m[key_molecule_index] for m in data.smiles()]),
            sampler="kmeans",
            hopts=dict(metric="jaccard"),
            fingerprint="morgan_fingerprint",
            fprints_hopts=dict(n_bits=2048),
            **astartes_kwargs,
        )
        train, val, test = _unpack_astartes_result(data, result, include_val)

    elif split == "kennard_stone":
        result = mol_split_fun(
            np.array([m[key_molecule_index] for m in data.smiles()]),
            sampler="kennard_stone",
            hopts=dict(metric="jaccard"),
            fingerprint="morgan_fingerprint",
            fprints_hopts=dict(n_bits=2048),
            **astartes_kwargs,
        )
        train, val, test = _unpack_astartes_result(data, result, include_val)

    else:
        raise ValueError(f'split type "{split}" not supported.')

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


def _unpack_astartes_result(data, result, include_val, log_stats=False):
    """Helper function to partition input data based on output of astartes sampler

    Args:
        data (MoleculeDataset): The data being partitioned
        result (tuple): Output from call to astartes containing the split indices
        include_val (bool): True if a validation set is included, False otherwise.
        log_stats (bool, optional): Print stats about scaffolds. Defaults to False.

    Returns:
        MoleculeDatset: The train, validation (can be empty) and test dataset
    """
    train_idxs, val_idxs, test_idxs = [], [], []
    if include_val:
        train_idxs, val_idxs, test_idxs = result[3], result[4], result[5]
    else:
        train_idxs, test_idxs = result[2], result[3]
    if log_stats:
        log_scaffold_stats(data, [set(train_idxs), set(val_idxs), set(test_idxs)])
    train = [data[i] for i in train_idxs]
    val = [data[i] for i in val_idxs]
    test = [data[i] for i in test_idxs]
    return train, val, test
