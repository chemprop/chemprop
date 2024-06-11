from collections.abc import Iterable, Sequence
import copy
from enum import auto
import logging

from astartes import train_test_split, train_val_test_split
from astartes.molecules import train_test_split_molecules, train_val_test_split_molecules
import numpy as np
from rdkit import Chem

from chemprop.data.datapoints import MoleculeDatapoint, ReactionDatapoint
from chemprop.utils.utils import EnumMapping

logger = logging.getLogger(__name__)

Datapoints = Sequence[MoleculeDatapoint] | Sequence[ReactionDatapoint]
MulticomponentDatapoints = Sequence[Datapoints]


class SplitType(EnumMapping):
    CV_NO_VAL = auto()
    CV = auto()
    SCAFFOLD_BALANCED = auto()
    RANDOM_WITH_REPEATED_SMILES = auto()
    RANDOM = auto()
    KENNARD_STONE = auto()
    KMEANS = auto()


def make_split_indices(
    mols: Sequence[Chem.Mol],
    split: SplitType | str = "random",
    sizes: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    num_folds: int = 1,
):
    """Splits data into training, validation, and test splits.

    Parameters
    ----------
    mols : Sequence[Chem.Mol]
        Sequence of RDKit molecules to use for structure based splitting
    split : SplitType | str, optional
        Split type, one of ~chemprop.data.utils.SplitType, by default "random"
    sizes : tuple[float, float, float], optional
        3-tuple with the proportions of data in the train, validation, and test sets, by default
        (0.8, 0.1, 0.1). Set the middle value to 0 for a two way split.
    seed : int, optional
        The random seed passed to astartes, by default 0
    num_folds : int, optional
        Number of folds to create (only needed for "cv" and "cv-no-test"), by default 1

    Returns
    -------
    tuple[list[int], list[int], list[int]] | tuple[list[list[int], ...], list[list[int], ...], list[list[int], ...]]
        A tuple of list of indices corresponding to the train, validation, and test splits of the
        data. If the split type is "cv" or "cv-no-test", returns a tuple of lists of lists of
        indices corresponding to the train, validation, and test splits of each fold.

        .. important::
            Validation may or may not be present

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

    n_datapoints = len(mols)
    train, val, test = None, None, None
    match SplitType.get(split):
        case SplitType.CV_NO_VAL | SplitType.CV:
            min_folds = 2 if SplitType.get(split) == SplitType.CV_NO_VAL else 3
            if not (min_folds <= num_folds <= n_datapoints):
                raise ValueError(
                    f"invalid number of folds requested! got: {num_folds}, but expected between "
                    f"{min_folds} and {n_datapoints} (i.e., number of datapoints), inclusive, "
                    f"for split type: {repr(split)}"
                )

            # returns nested lists of indices
            train, val, test = [], [], []
            random = np.random.default_rng(seed)

            indices = np.tile(np.arange(num_folds), 1 + n_datapoints // num_folds)[:n_datapoints]
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
            for mol in mols:
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
            all_smiles = np.array([Chem.MolToSmiles(mol) for mol in mols])
            unique_smiles = np.unique(all_smiles)

            # save a mapping of smiles -> all the indices that it appeared at
            smiles_indices = {}
            for smiles in unique_smiles:
                smiles_indices[smiles] = np.where(all_smiles == smiles)[0].tolist()

            # randomly split the unique smiles
            result = split_fun(np.arange(len(unique_smiles)), sampler="random", **astartes_kwargs)
            train_idxs, val_idxs, test_idxs = _unpack_astartes_result(result, include_val)

            # convert these to the 'actual' indices from the original list using the dict we made
            train = sum((smiles_indices[unique_smiles[i]] for i in train_idxs), [])
            val = sum((smiles_indices[unique_smiles[j]] for j in val_idxs), [])
            test = sum((smiles_indices[unique_smiles[k]] for k in test_idxs), [])

        case SplitType.RANDOM:
            result = split_fun(np.arange(n_datapoints), sampler="random", **astartes_kwargs)
            train, val, test = _unpack_astartes_result(result, include_val)

        case SplitType.KENNARD_STONE:
            result = mol_split_fun(
                np.array(mols),
                sampler="kennard_stone",
                hopts=dict(metric="jaccard"),
                fingerprint="morgan_fingerprint",
                fprints_hopts=dict(n_bits=2048),
                **astartes_kwargs,
            )
            train, val, test = _unpack_astartes_result(result, include_val)

        case SplitType.KMEANS:
            result = mol_split_fun(
                np.array(mols),
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
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
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
    .. important::
        validation possibly empty
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


def split_data_by_indices(
    data: Datapoints | MulticomponentDatapoints,
    train_indices: Iterable[Iterable[int]] | Iterable[int] | None = None,
    val_indices: Iterable[Iterable[int]] | Iterable[int] | None = None,
    test_indices: Iterable[Iterable[int]] | Iterable[int] | None = None,
):
    """Splits data into training, validation, and test groups based on split indices given."""

    train_data = _splitter_helper(data, train_indices) if train_indices is not None else None
    val_data = _splitter_helper(data, val_indices) if val_indices is not None else None
    test_data = _splitter_helper(data, test_indices) if test_indices is not None else None

    return train_data, val_data, test_data


def _splitter_helper(data, indices):
    nested_component = not isinstance(data[0], (MoleculeDatapoint, ReactionDatapoint))
    nested_split = isinstance(indices[0], Iterable)

    match (nested_component, nested_split):
        case (False, False):
            datapoints = data
            idxs = indices
            return [datapoints[idx] for idx in idxs]
        case (False, True):
            datapoints = data
            idxss = indices
            return [[datapoints[idx] for idx in idxs] for idxs in idxss]
        case (True, False):
            datapointss = data
            idxs = indices
            return [[datapoints[idx] for idx in idxs] for datapoints in datapointss]
        case (True, True):
            datapointss = data
            idxss = indices
            return [
                [[datapoints[idx] for idx in idxs] for datapoints in datapointss] for idxs in idxss
            ]
