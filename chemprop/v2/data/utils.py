import copy
import itertools
import logging
from enum import auto
from typing import Sequence

import numpy as np
from astartes import train_test_split, train_val_test_split
from astartes.molecules import train_test_split_molecules, train_val_test_split_molecules
from rdkit import Chem

from chemprop.v2.data.datapoints import MoleculeDatapoint
from chemprop.v2.utils.utils import EnumMapping

logger = logging.getLogger(__name__)


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
) -> tuple[Sequence[MoleculeDatapoint], ...]:
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
    tuple[Sequence[MoleculeDatapoint], ...]
        A tuple of Sequences of ~chemprop.data.MoleculeDatapoint containing the train, validation, and test splits of the data.
            NOTE: validation may or may not be present

    Raises
    ------
    RuntimeError
        Requested split sizes tuple not of length 3
    ValueError
        Innapropriate number of folds requested
    ValueError
        Unsupported split method requested
    """
    if (num_splits := len(sizes)) != 3:
        raise RuntimeError(
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
        case SplitType.CV_NO_VAL, SplitType.CV:
            if (max_folds := len(datapoints)) > num_folds or num_folds <= 1:
                raise ValueError(
                    f"Number of folds for cross-validation must be between 2 and {max_folds} (length of data) inclusive (got {num_folds})."
                )

            train, val, test = [], [], []
            for _ in range(len(num_folds)):
                result = split_fun(np.arange(len(datapoints)), sampler="random", **astartes_kwargs)
                i_train, i_val, i_test = _unpack_astartes_result(datapoints, result, include_val)
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
            train, val, test = _unpack_astartes_result(datapoints, result, include_val)

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
            train_idxs, val_idxs, test_idxs = _unpack_astartes_result(None, result, include_val)

            # convert these to the 'actual' indices from the original list using the dict we made
            train = [
                datapoints[ii]
                for ii in itertools.chain.from_iterable(
                    smiles_indices[unique_smiles[i]] for i in train_idxs
                )
            ]
            val = [
                datapoints[ii]
                for ii in itertools.chain.from_iterable(
                    smiles_indices[unique_smiles[i]] for i in val_idxs
                )
            ]
            test = [
                datapoints[ii]
                for ii in itertools.chain.from_iterable(
                    smiles_indices[unique_smiles[i]] for i in test_idxs
                )
            ]

        case SplitType.RANDOM:
            result = split_fun(np.arange(len(datapoints)), sampler="random", **astartes_kwargs)
            train, val, test = _unpack_astartes_result(datapoints, result, include_val)

        case SplitType.KENNARD_STONE:
            result = mol_split_fun(
                np.array([d.mol for d in datapoints]),
                sampler="kennard_stone",
                hopts=dict(metric="jaccard"),
                fingerprint="morgan_fingerprint",
                fprints_hopts=dict(n_bits=2048),
                **astartes_kwargs,
            )
            train, val, test = _unpack_astartes_result(datapoints, result, include_val)

        case SplitType.KMEANS:
            result = mol_split_fun(
                np.array([d.mol for d in datapoints]),
                sampler="kmeans",
                hopts=dict(metric="jaccard"),
                fingerprint="morgan_fingerprint",
                fprints_hopts=dict(n_bits=2048),
                **astartes_kwargs,
            )
            train, val, test = _unpack_astartes_result(datapoints, result, include_val)

        case _:
            raise ValueError(f'split type "{split}" not supported.')

    return train, val, test


def _unpack_astartes_result(
    data: Sequence[MoleculeDatapoint], result: tuple, include_val: bool
) -> tuple[list[MoleculeDatapoint], list[MoleculeDatapoint], list[MoleculeDatapoint]]:
    """Helper function to partition input data based on output of astartes sampler

    Parameters
    -----------
    data: MoleculeDataset
        The data being partitioned. If None, returns indices.
    result: tuple
        Output from call to astartes containing the split indices
    include_val: bool
        True if a validation set is included, False otherwise.

    Returns
    ---------
    train: MoleculeDataset
    val: MoleculeDataset
        NOTE: possibly empty
    test: MoleculeDataset
    """
    train_idxs, val_idxs, test_idxs = [], [], []
    # astartes returns a set of lists containing the data, clusters (if applicable)
    # and indices (always last), so we pull out the indices
    if include_val:
        train_idxs, val_idxs, test_idxs = result[-3], result[-2], result[-1]
    else:
        train_idxs, test_idxs = result[-2], result[-1]
    if data is None:
        return train_idxs, val_idxs, test_idxs
    train = [data[i] for i in train_idxs]
    val = [data[i] for i in val_idxs]
    test = [data[i] for i in test_idxs]
    return train, val, test
