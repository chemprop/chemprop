import copy
import itertools
import logging
from enum import auto
from typing import Sequence
from random import Random
import numpy as np
from astartes import train_test_split, train_val_test_split
from astartes.molecules import train_test_split_molecules, train_val_test_split_molecules
from rdkit import Chem

from chemprop.data.datapoints import MoleculeDatapoint
from chemprop.utils.utils import EnumMapping

logger = logging.getLogger(__name__)
MulticomponentDatapoint = Sequence[MoleculeDatapoint]


class SplitType(EnumMapping):
    CV_NO_VAL = auto()
    CV = auto()
    SCAFFOLD_BALANCED = auto()
    RANDOM_WITH_REPEATED_SMILES = auto()
    RANDOM = auto()
    KENNARD_STONE = auto()
    KMEANS = auto()


def split_data(
    datapoints: Sequence[MoleculeDatapoint],
    split: SplitType | str = "random",
    sizes: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    num_folds: int = 1,
):
    """Splits data into training, validation, and test splits.

    Parameters
    ----------
    datapoints : Sequence[MoleculeDatapoint]
        Sequence of chemprop.data.MoleculeDatapoint.
    split : SplitType | str, optional
        Split type, one of ~chemprop.data.utils.SplitType, by default "random"
    sizes : tuple[float, float, float], optional
        3-tuple with the proportions of data in the train, validation, and test sets, by default (0.8, 0.1, 0.1)
    seed : int, optional
        The random seed passed to astartes, by default 0
    num_folds : int, optional
        Number of folds to create (only needed for "cv" and "cv-no-test"), by default 1

    Returns
    -------
    tuple[list[int], list[int], list[int]] | tuple[list[list[int], ...], list[list[int], ...], list[list[int], ...]]
        A tuple of list of indices corresponding to the train, validation, and test splits of the data.
        If the split type is "cv" or "cv-no-test", returns a tuple of lists of lists of indices corresponding to the train, validation, and test splits of each fold.
            NOTE: validation may or may not be present

    Raises
    ------
    ValueError
        Requested split sizes tuple not of length 3
    ValueError
        Innapropriate number of folds requested
    ValueError
        Unsupported split method requested
    """
    if (num_splits := len(sizes)) != 3:
        raise ValueError(
            f"Specify sizes for train, validation, and test (got {num_splits} values)."
        )
    # typically include a validation set
    include_val = True
    split_fun = train_val_test_split
    mol_split_fun = train_val_test_split_molecules
    # default sampling arguments for astartes sampler
    astartes_kwargs = dict(
        train_size=sizes[0], test_size=sizes[2], return_indices=True, random_state=seed
    )
    # if no validation set, reassign the splitting functions
    if sizes[1] == 0.0:
        include_val = False
        split_fun = train_test_split
        mol_split_fun = train_test_split_molecules
    else:
        astartes_kwargs["val_size"] = sizes[1]

    train, val, test = None, None, None
    match SplitType.get(split):
        case SplitType.CV_NO_VAL | SplitType.CV:
            if not (2 <= num_folds <= len(datapoints)):
                min_folds = 2 if SplitType.get(split) == SplitType.CV_NO_VAL else 3
                raise ValueError(
                    f"invalid number of folds requested! got: {num_folds}, but expected between {min_folds} and {len(datapoints)} (i.e., number of datapoints), inclusive, for split type: {repr(split)}"
                )

            # returns nested lists of indices
            train, val, test = [], [], []
            random = np.random.default_rng(seed)

            indices = np.tile(np.arange(num_folds), 1 + len(datapoints) // num_folds)[
                : len(datapoints)
            ]
            random.shuffle(indices)

            for fold_idx in range(num_folds):
                test_index = fold_idx
                val_index = (fold_idx + 1) % num_folds

                if split != SplitType.CV_NO_VAL:
                    i_val = np.where(indices == val_index)[0]
                    i_test = np.where(indices == test_index)[0]
                    i_train = np.where((indices != val_index) & (indices != test_index))[0]
                else:
                    i_val = []
                    i_test = np.where(indices == test_index)[0]
                    i_train = np.where(indices != test_index)[0]

                train.append(i_train)
                val.append(i_val)
                test.append(i_test)

        case SplitType.SCAFFOLD_BALANCED:
            mols_without_atommaps = []
            for d in datapoints:
                mol = d.mol
                copied_mol = copy.deepcopy(mol)
                for atom in copied_mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                mols_without_atommaps.append(copied_mol)
            result = mol_split_fun(
                np.array(mols_without_atommaps), sampler="scaffold", **astartes_kwargs
            )
            train, val, test = _unpack_astartes_result(result, include_val)

        # Use to constrain data with the same smiles go in the same split.
        case SplitType.RANDOM_WITH_REPEATED_SMILES:
            # get two arrays: one of all the smiles strings, one of just the unique
            all_smiles = np.array([Chem.MolToSmiles(d.mol) for d in datapoints])
            unique_smiles = np.unique(all_smiles)

            # save a mapping of smiles -> all the indices that it appeared at
            smiles_indices = {}
            for smiles in unique_smiles:
                smiles_indices[smiles] = np.where(all_smiles == smiles)[0]

            # randomly split the unique smiles
            result = split_fun(np.arange(len(unique_smiles)), sampler="random", **astartes_kwargs)
            train_idxs, val_idxs, test_idxs = _unpack_astartes_result(result, include_val)

            # convert these to the 'actual' indices from the original list using the dict we made
            train = list(
                itertools.chain.from_iterable(smiles_indices[unique_smiles[i]] for i in train_idxs)
            )
            val = list(
                itertools.chain.from_iterable(smiles_indices[unique_smiles[i]] for i in val_idxs)
            )
            test = list(
                itertools.chain.from_iterable(smiles_indices[unique_smiles[i]] for i in test_idxs)
            )

        case SplitType.RANDOM:
            result = split_fun(np.arange(len(datapoints)), sampler="random", **astartes_kwargs)
            train, val, test = _unpack_astartes_result(result, include_val)

        case SplitType.KENNARD_STONE:
            result = mol_split_fun(
                np.array([d.mol for d in datapoints]),
                sampler="kennard_stone",
                hopts=dict(metric="jaccard"),
                fingerprint="morgan_fingerprint",
                fprints_hopts=dict(n_bits=2048),
                **astartes_kwargs,
            )
            train, val, test = _unpack_astartes_result(result, include_val)

        case SplitType.KMEANS:
            result = mol_split_fun(
                np.array([d.mol for d in datapoints]),
                sampler="kmeans",
                hopts=dict(metric="jaccard"),
                fingerprint="morgan_fingerprint",
                fprints_hopts=dict(n_bits=2048),
                **astartes_kwargs,
            )
            train, val, test = _unpack_astartes_result(result, include_val)

        case _:
            raise RuntimeError("Unreachable code reached!")

    return train, val, test


def _unpack_astartes_result(
    result: tuple, include_val: bool
) -> tuple[list[int], list[int], list[int]]:
    """Helper function to partition input data based on output of astartes sampler

    Parameters
    -----------
    result: tuple
        Output from call to astartes containing the split indices
    include_val: bool
        True if a validation set is included, False otherwise.

    Returns
    ---------
    train: list[int]
    val: list[int]
        NOTE: possibly empty
    test: list[int]
    """
    train_idxs, val_idxs, test_idxs = [], [], []
    # astartes returns a set of lists containing the data, clusters (if applicable)
    # and indices (always last), so we pull out the indices
    if include_val:
        train_idxs, val_idxs, test_idxs = result[-3], result[-2], result[-1]
    else:
        train_idxs, test_idxs = result[-2], result[-1]
    return list(train_idxs), list(val_idxs), list(test_idxs)


def split_component(
    datapointss: Sequence[Sequence[MoleculeDatapoint | MulticomponentDatapoint]],
    split: SplitType | str = "random",
    key_index: int = 0,
    **kwargs,
):
    """Splits multicomponent data into training, validation, and test splits."""

    key_datapoints = datapointss[key_index]
    train_idxs, val_idxs, test_idxs = split_data(key_datapoints, split=split, **kwargs)

    match SplitType.get(split):
        case SplitType.CV_NO_VAL | SplitType.CV:
            # convert indices to datapoints for each fold
            train = [
                [[datapoints[i] for i in fold] for datapoints in datapointss] for fold in train_idxs
            ]
            val = [
                [[datapoints[i] for i in fold] for datapoints in datapointss] for fold in val_idxs
            ]
            test = [
                [[datapoints[i] for i in fold] for datapoints in datapointss] for fold in test_idxs
            ]
        case SplitType.SCAFFOLD_BALANCED | SplitType.RANDOM_WITH_REPEATED_SMILES | SplitType.RANDOM | SplitType.KENNARD_STONE | SplitType.KMEANS:
            # convert indices to datapoints
            train = [[datapoints[i] for i in train_idxs] for datapoints in datapointss]
            val = [[datapoints[i] for i in val_idxs] for datapoints in datapointss]
            test = [[datapoints[i] for i in test_idxs] for datapoints in datapointss]
        case _:
            raise RuntimeError("Unreachable code reached!")

    return train, val, test
