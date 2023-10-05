import copy
import itertools
import logging
from enum import auto
from typing import Sequence, Tuple

import numpy as np
from astartes import train_test_split, train_val_test_split
from astartes.molecules import train_test_split_molecules, train_val_test_split_molecules

from chemprop.v2.cli.utils import LOG_DIR, NOW
from chemprop.v2.data.datapoints import MoleculeDatapoint
from chemprop.v2.data.datasets import MoleculeDataset
from chemprop.v2.data.utils import log_scaffold_stats
from chemprop.v2.utils.utils import AutoName

logger = logging.getLogger(__name__)

DEFAULT_LOGFILE = f"{LOG_DIR}/{NOW}.log"
LOG_LEVELS = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
SUBCOMMANDS = []


class SplitType(AutoName):
    CV_NO_VAL = auto()
    CV = auto()
    SCAFFOLD_BALANCED = auto()
    RANDOM_WITH_REPEATED_SMILES = auto()
    RANDOM = auto()
    KENNARD_STONE = auto()
    KMEANS = auto()


def split_data(
    data: Sequence[MoleculeDatapoint],
    split: str | SplitType,
    sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    key_molecule_index: int = 0,
    seed: int = 0,
    num_folds: int = 1,
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
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Invalid train/val/test splits! got: {sizes}")

    # typically include a validation set
    include_val = True
    split_fun = train_val_test_split
    mol_split_fun = train_val_test_split_molecules
    # default sampling arguments for astartes sampler
    astartes_kwargs = dict(train_size=sizes[0], test_size=sizes[2], return_indices=True, random_state=seed)
    # if no validation set, reassign the splitting functions
    if not sizes[1]:
        include_val = False
        split_fun = train_test_split
        mol_split_fun = train_test_split_molecules
    else:
        astartes_kwargs["val_size"] = sizes[1]

    train, val, test = None, None, None
    if split in {SplitType.CV_NO_VAL, SplitType.CV}:
        if (max_folds := len(data)) > num_folds or num_folds <= 1:
            raise ValueError(f"Number of folds for cross-validation must be between 2 and {max_folds} (length of data) inclusive (got {num_folds}).")

        train, val, test = [], [], []
        for _ in range(len(num_folds)):
            result = split_fun(np.arange(len(data)), sampler="random", **astartes_kwargs)
            i_train, i_val, i_test = _unpack_astartes_result(data, result, include_val)
            train.append(i_train)
            val.append(i_val)
            test.append(i_test)

    elif split == SplitType.SCAFFOLD_BALANCED:
        mols_without_atommaps = []
        for mol in data.mols(flatten=False):
            copied_mol = copy.deepcopy(mol[key_molecule_index])
            for atom in copied_mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mols_without_atommaps.append([copied_mol])
        result = mol_split_fun(np.array(mols_without_atommaps), sampler="scaffold", **astartes_kwargs)
        train, val, test = _unpack_astartes_result(data, result, include_val, log_stats=True)

    # Use to constrain data with the same smiles go in the same split.
    elif split == SplitType.RANDOM_WITH_REPEATED_SMILES:
        # get two arrays: one of all the smiles strings, one of just the unique
        all_smiles = np.array(data.smiles())
        unique_smiles = np.unique(all_smiles)

        # save a mapping of smiles -> all the indices that it appeared at
        smiles_indices = {}
        for smiles in unique_smiles:
            smiles_indices[smiles] = np.where(all_smiles == smiles)[0]

        # randomly split the unique smiles
        result = split_fun(np.arange(len(unique_smiles)))
        train_idxs, val_idxs, test_idxs = _unpack_astartes_result(None, result, include_val)

        # convert these to the 'actual' indices from the original list using the dict we made
        train = list(itertools.chain.from_iterable(smiles_indices[unique_smiles[i]] for i in train_idxs))
        val = list(itertools.chain.from_iterable(smiles_indices[unique_smiles[i]] for i in val_idxs))
        test = list(itertools.chain.from_iterable(smiles_indices[unique_smiles[i]] for i in test_idxs))

    elif split == SplitType.RANDOM:
        result = split_fun(np.arange(len(data)), sampler="random", **astartes_kwargs)
        train, val, test = _unpack_astartes_result(data, result, include_val)

    elif split == SplitType.KMEANS:
        result = mol_split_fun(
            np.array([m[key_molecule_index] for m in data.smiles()]),
            sampler="kmeans",
            hopts=dict(metric="jaccard"),
            fingerprint="morgan_fingerprint",
            fprints_hopts=dict(n_bits=2048),
            **astartes_kwargs,
        )
        train, val, test = _unpack_astartes_result(data, result, include_val)

    elif split == SplitType.KENNARD_STONE:
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
        data (MoleculeDataset): The data being partitioned. If None, returns indices.
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
    if data is None:
        return train_idxs, val_idxs, test_idxs
    train = [data[i] for i in train_idxs]
    val = [data[i] for i in val_idxs]
    test = [data[i] for i in test_idxs]
    return train, val, test
