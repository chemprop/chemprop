<<<<<<< HEAD
from collections import OrderedDict, defaultdict
import csv
from logging import Logger
import pickle
from random import Random
from typing import List, Set, Tuple, Union
import os

from rdkit import Chem
import numpy as np
from tqdm import tqdm

from .datasets import MoleculeDataset, ReactionDataset
from .datapoints import MoleculeDatapoint, ReactionDatapoint
from .data import make_mols
# from .scaffold import log_scaffold_stats, scaffold_split
from chemprop.args import PredictArgs, TrainArgs
from chemprop.featurizers import load_features, load_valid_atom_or_bond_features, is_mol

def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def preprocess_smiles_columns(path: str,
                              smiles_columns: Union[str, List[str]] = None,
                              number_of_molecules: int = 1) -> List[str]:
    """
    Preprocesses the :code:`smiles_columns` variable to ensure that it is a list of column
    headings corresponding to the columns in the data file holding SMILES. Assumes file has a header.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules with associated SMILES for each
                           data point.
    :return: The preprocessed version of :code:`smiles_columns` which is guaranteed to be a list.
    """

    if smiles_columns is None:
        if os.path.isfile(path):
            columns = get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None]*number_of_molecules
    else:
        if not isinstance(smiles_columns,list):
            smiles_columns=[smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError('Length of smiles_columns must match number_of_molecules.')
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError('Provided smiles_columns do not match the header of data file.')

    return smiles_columns


def get_task_names(path: str,
                   smiles_columns: Union[str, List[str]] = None,
                   target_columns: List[str] = None,
                   ignore_columns: List[str] = None) -> List[str]:
    """
    Gets the task names from a data CSV file.

    If :code:`target_columns` is provided, returns `target_columns`.
    Otherwise, returns all columns except the :code:`smiles_columns`
    (or the first column, if the :code:`smiles_columns` is None) and
    the :code:`ignore_columns`.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_columns` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :return: A list of task names.
    """
    if target_columns is not None:
        return target_columns

    columns = get_header(path)

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    ignore_columns = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))

    target_names = [column for column in columns if column not in ignore_columns]

    return target_names


def get_data_weights(path: str) -> List[float]:
    """
    Returns the list of data weights for the loss function as stored in a CSV file.

    :param path: Path to a CSV file.
    :return: A list of floats containing the data weights.
    """
    weights = []
    with open(path) as f:
        reader=csv.reader(f)
        next(reader) #skip header row
        for line in reader:
            weights.append(float(line[0]))
    # normalize the data weights
    avg_weight=sum(weights)/len(weights)
    weights = [w/avg_weight for w in weights]
    if min(weights) < 0:
        raise ValueError('Data weights must be non-negative for each datapoint.')
    return weights


def get_smiles(path: str,
               smiles_columns: Union[str, List[str]] = None,
               number_of_molecules: int = 1,
               header: bool = True,
               flatten: bool = False
               ) -> Union[List[str], List[List[str]]]:
    """
    Returns the SMILES from a data CSV file.

    :param path: Path to a CSV file.
    :param smiles_columns: A list of the names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules for each data point. Not necessary if
                                the names of smiles columns are previously processed.
    :param header: Whether the CSV file contains a header.
    :param flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
    :return: A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
    """
    if smiles_columns is not None and not header:
        raise ValueError('If smiles_column is provided, the CSV file must have a header.')

    if not isinstance(smiles_columns, list) and header:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns, number_of_molecules=number_of_molecules)

    with open(path) as f:
        if header:
            reader = csv.DictReader(f)
        else:
            reader = csv.reader(f)
            smiles_columns = list(range(number_of_molecules))

        smiles = [[row[c] for c in smiles_columns] for row in reader]

    if flatten:
        smiles = [smile for smiles_list in smiles for smile in smiles_list]

    return smiles


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :return: A :class:`~chemprop.data.MoleculeDataset` with only the valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in tqdm(data)
                            if all(s != '' for s in datapoint.smiles) and all(m is not None for m in datapoint.mol)
                            and all(m.GetNumHeavyAtoms() > 0 for m in datapoint.mol if not isinstance(m, tuple))
                            and all(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() > 0 for m in datapoint.mol if isinstance(m, tuple))])


def get_invalid_smiles_from_file(path: str = None,
               smiles_columns: Union[str, List[str]] = None,
               header: bool = True,
               reaction: bool = False,
               ) -> Union[List[str], List[List[str]]]:
    """
    Returns the invalid SMILES from a data CSV file.

    :param path: Path to a CSV file.
    :param smiles_columns: A list of the names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param header: Whether the CSV file contains a header.
    :param reaction: Boolean whether the SMILES strings are to be treated as a reaction.
    :return: A list of lists of SMILES, for the invalid SMILES in the file.
    """
    smiles = get_smiles(path=path, smiles_columns=smiles_columns, header=header)

    invalid_smiles = get_invalid_smiles_from_list(smiles=smiles, reaction=reaction)

    return invalid_smiles


def get_invalid_smiles_from_list(smiles: List[List[str]], reaction: bool = False) -> List[List[str]]:
    """
    Returns the invalid SMILES from a list of lists of SMILES strings.

    :param smiles: A list of list of SMILES.
    :param reaction: Boolean whether the SMILES strings are to be treated as a reaction.
    :return: A list of lists of SMILES, for the invalid SMILES among the lists provided.
    """
    invalid_smiles = []

    # If the first SMILES in the column is a molecule, the remaining SMILES in the same column should all be a molecule.
    # Similarly, if the first SMILES in the column is a reaction, the remaining SMILES in the same column should all
    # correspond to reaction. Therefore, get `is_mol_list` only using the first element in smiles.
    is_mol_list = [is_mol(s) for s in smiles[0]]
    is_reaction_list = [True if not x and reaction else False for x in is_mol_list]
    is_explicit_h_list = [False for x in is_mol_list]  # set this to False as it is not needed for invalid SMILES check
    is_adding_hs_list = [False for x in is_mol_list]  # set this to False as it is not needed for invalid SMILES check

    for mol_smiles in smiles:
        mols = make_mols(smiles=mol_smiles, reaction_list=is_reaction_list, keep_h_list=is_explicit_h_list,
                         add_h_list=is_adding_hs_list)
        if any(s == '' for s in mol_smiles) or \
           any(m is None for m in mols) or \
           any(m.GetNumHeavyAtoms() == 0 for m in mols if not isinstance(m, tuple)) or \
           any(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() == 0 for m in mols if isinstance(m, tuple)):

            invalid_smiles.append(mol_smiles)

    return invalid_smiles


def get_data(path: str,
             smiles_columns: Union[str, List[str]] = None,
             target_columns: List[str] = None,
             ignore_columns: List[str] = None,
             skip_invalid_smiles: bool = True,
             data_weights_path: str = None,
             features_path: List[str] = None,
             features_generator: List[str] = None,
             phase_features_path: str = None,
             atom_descriptors_path: str = None,
             bond_features_path: str = None,
             max_data_size: int = None,
             store_row: bool = False,
             logger: Logger = None,
             loss_function: str = None,
             skip_none_targets: bool = False,
             atom_descriptors_type: str = None,
             ) -> MoleculeDataset:
    """
    Gets SMILES and target values from a CSV file.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_column` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles using :func:`filter_invalid_smiles`.
    :param args: Arguments, either :class:`~chemprop.args.TrainArgs` or :class:`~chemprop.args.PredictArgs`.
    :param data_weights_path: A path to a file containing weights for each molecule in the loss function.
    :param features_path: A list of paths to files containing features. If provided, it is used
                          in place of :code:`args.features_path`.
    :param features_generator: A list of features generators to use. If provided, it is used
                               in place of :code:`args.features_generator`.
    :param phase_features_path: A path to a file containing phase features as applicable to spectra.
    :param atom_descriptors_path: The path to the file containing the custom atom descriptors.
    :param bond_features_path: The path to the file containing the custom bond features.
    :param max_data_size: The maximum number of data points to load.
    :param logger: A logger for recording output.
    :param store_row: Whether to store the raw CSV row in each :class:`~chemprop.data.data.MoleculeDatapoint`.
    :param skip_none_targets: Whether to skip targets that are all 'None'. This is mostly relevant when --target_columns
                              are passed in, so only a subset of tasks are examined.
    :param loss_function: The loss function to be used in training.
    :param atom_descriptors:
    :return: A :class:`~chemprop.data.MoleculeDataset` containing SMILES and target values along
             with other info such as additional features when desired.
    """
    debug = logger.debug if logger is not None else print

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    max_data_size = max_data_size or float('inf')

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None
        
    if phase_features_path is not None:
        phase_features = load_features(phase_features_path)
        for d_phase in phase_features:
            if not (d_phase.sum() == 1 and np.count_nonzero(d_phase) == 1):
                raise ValueError('Phase features must be one-hot encoded.')
        if features_data is not None:
            features_data = np.concatenate((features_data,phase_features), axis=1)
        else: # if there are no other molecular features, phase features become the only molecular features
            features_data = np.array(phase_features)
    else:
        phase_features = None

    # Load data weights
    if data_weights_path is not None:
        data_weights = get_data_weights(data_weights_path)
    else:
        data_weights = None

    # By default, the targets columns are all the columns except the SMILES column
    if target_columns is None:
        target_columns = get_task_names(
            path=path,
            smiles_columns=smiles_columns,
            target_columns=target_columns,
            ignore_columns=ignore_columns,
        )

    # Find targets provided as inequalities
    if loss_function == 'bounded_mse':
        gt_targets, lt_targets = get_inequality_targets(path=path, target_columns=target_columns)
    else:
        gt_targets, lt_targets = None, None

    # Load data
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if any([c not in fieldnames for c in smiles_columns]):
            raise ValueError(f'Data file did not contain all provided smiles columns: {smiles_columns}. Data file field names are: {fieldnames}')
        if any([c not in fieldnames for c in target_columns]):
            raise ValueError(f'Data file did not contain all provided target columns: {target_columns}. Data file field names are: {fieldnames}')

        all_smiles, all_targets, all_rows, all_features, all_phase_features, all_weights, all_gt, all_lt = [], [], [], [], [], [], [], []
        for i, row in enumerate(tqdm(reader)):
            smiles = [row[c] for c in smiles_columns]

            targets = []
            for column in target_columns:
                value = row[column]
                if value in ['','nan']:
                    targets.append(None)
                elif '>' in value or '<' in value:
                    if loss_function == 'bounded_mse':
                        targets.append(float(value.strip('<>')))
                    else:
                        raise ValueError('Inequality found in target data. To use inequality targets (> or <), the regression loss function bounded_mse must be used.')
                else:
                    targets.append(float(value))

            # Check whether all targets are None and skip if so
            if skip_none_targets and all(x is None for x in targets):
                continue

            all_smiles.append(smiles)
            all_targets.append(targets)

            if features_data is not None:
                all_features.append(features_data[i])
            
            if phase_features is not None:
                all_phase_features.append(phase_features[i])

            if data_weights is not None:
                all_weights.append(data_weights[i])

            if gt_targets is not None:
                all_gt.append(gt_targets[i])

            if lt_targets is not None:
                all_lt.append(lt_targets[i])

            if store_row:
                all_rows.append(row)

            if len(all_smiles) >= max_data_size:
                break

        atom_features = None
        atom_descriptors = None
        if atom_descriptors_type is not None:
            try:
                descriptors = load_valid_atom_or_bond_features(atom_descriptors_path, [x[0] for x in all_smiles])
            except Exception as e:
                raise ValueError(f'Failed to load or validate custom atomic descriptors or features: {e}')

            if atom_descriptors_type == 'feature':
                atom_features = descriptors
            elif atom_descriptors_type == 'descriptor':
                atom_descriptors = descriptors

        bond_features = None
        if bond_features_path is not None:
            try:
                bond_features = load_valid_atom_or_bond_features(bond_features_path, [x[0] for x in all_smiles])
            except Exception as e:
                raise ValueError(f'Failed to load or validate custom bond features: {e}')
        
        # Check if this data is for a reaction (multiple SMILES)
        if len(smiles_columns) > 1:
            data = ReactionDataset([
                ReactionDatapoint(
                    smis=smiles,
                    targets=targets,
                    row=all_rows[i] if store_row else None,
                    data_weight=all_weights[i] if data_weights is not None else None,
                    gt_targets=all_gt[i] if gt_targets is not None else None,
                    lt_targets=all_lt[i] if lt_targets is not None else None,
                    features_generators=features_generator,
                    features=all_features[i] if features_data is not None else None,
                    phase_features=all_phase_features[i] if phase_features is not None else None,
                    atom_features=atom_features[i] if atom_features is not None else None,
                    atom_descriptors=atom_descriptors[i] if atom_descriptors is not None else None,
                    bond_features=bond_features[i] if bond_features is not None else None,
                    keep_h = False,
                    add_h = False
                ) for i, (smiles, targets) in tqdm(enumerate(zip(all_smiles, all_targets)),
                                                total=len(all_smiles))
            ])
        elif len(smiles_columns) == 1:
            data = MoleculeDataset([
                MoleculeDatapoint(
                    smi=smiles[0],
                    targets=targets,
                    row=all_rows[i] if store_row else None,
                    data_weight=all_weights[i] if data_weights is not None else None,
                    gt_targets=all_gt[i] if gt_targets is not None else None,
                    lt_targets=all_lt[i] if lt_targets is not None else None,
                    features_generators=features_generator,
                    features=all_features[i] if features_data is not None else None,
                    phase_features=all_phase_features[i] if phase_features is not None else None,
                    atom_features=atom_features[i] if atom_features is not None else None,
                    atom_descriptors=atom_descriptors[i] if atom_descriptors is not None else None,
                    bond_features=bond_features[i] if bond_features is not None else None,
                    keep_h = False,
                    add_h = False
                ) for i, (smiles, targets) in tqdm(enumerate(zip(all_smiles, all_targets)),
                                                total=len(all_smiles))
            ], None)

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def get_data_from_smiles(smiles: List[List[str]],
                         skip_invalid_smiles: bool = True,
                         logger: Logger = None,
                         features_generator: List[str] = None) -> MoleculeDataset:
    """
    Converts a list of SMILES to a :class:`~chemprop.data.MoleculeDataset`.

    :param smiles: A list of lists of SMILES with length depending on the number of molecules.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles using :func:`filter_invalid_smiles`
    :param logger: A logger for recording output.
    :param features_generator: List of features generators.
    :return: A :class:`~chemprop.data.MoleculeDataset` with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=smile,
            row=OrderedDict({'smiles': smile}),
            features_generator=features_generator
        ) for smile in smiles
    ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def get_inequality_targets(path: str, target_columns: List[str] = None) -> List[str]:
    """

    """
    gt_targets = []
    lt_targets = []

    with open(path) as f:
        reader = csv.DictReader(f)
        for line in reader:
            values = [line[col] for col in target_columns]
            gt_targets.append(['>' in val for val in values])
            lt_targets.append(['<' in val for val in values])
            if any(['<' in val and '>' in val for val in values]):
                raise ValueError(f'A target value in csv file {path} contains both ">" and "<" symbols. Inequality targets must be on one edge and not express a range.')

    return gt_targets, lt_targets

def get_class_sizes(data: MoleculeDataset, proportion: bool = True) -> List[List[float]]:
    """
    Determines the proportions of the different classes in a classification dataset.

    :param data: A classification :class:`~chemprop.data.MoleculeDataset`.
    :param proportion: Choice of whether to return proportions for class size or counts.
    :return: A list of lists of class proportions. Each inner list contains the class proportions for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if targets[i][task_num] is not None:
                valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        if set(np.unique(task_targets)) > {0, 1}:
            raise ValueError('Classification dataset must only contains 0s and 1s.')
        if proportion:
            try:
                ones = np.count_nonzero(task_targets) / len(task_targets)
            except ZeroDivisionError:
                ones = float('nan')
                print('Warning: class has no targets')
            class_sizes.append([1 - ones, ones])
        else: # counts
            ones = np.count_nonzero(task_targets)
            class_sizes.append([len(task_targets) - ones, ones])

    return class_sizes


#  TODO: Validate multiclass dataset type.
def validate_dataset_type(data: MoleculeDataset, dataset_type: str) -> None:
    """
    Validates the dataset type to ensure the data matches the provided type.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :param dataset_type: The dataset type to check.
    """
    target_set = {target for targets in data.targets() for target in targets} - {None}
    classification_target_set = {0, 1}

    if dataset_type == 'classification' and not (target_set <= classification_target_set):
        raise ValueError('Classification data targets must only be 0 or 1 (or None). '
                         'Please switch to regression.')
    elif dataset_type == 'regression' and target_set <= classification_target_set:
        raise ValueError('Regression data targets must be more than just 0 or 1 (or None). '
                         'Please switch to classification.')


def validate_data(data_path: str) -> Set[str]:
    """
    Validates a data CSV file, returning a set of errors.

    :param data_path: Path to a data CSV file.
    :return: A set of error messages.
    """
    errors = set()

    header = get_header(data_path)

    with open(data_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        smiles, targets = [], []
        for line in reader:
            smiles.append(line[0])
            targets.append(line[1:])

    # Validate header
    if len(header) == 0:
        errors.add('Empty header')
    elif len(header) < 2:
        errors.add('Header must include task names.')

    mol = Chem.MolFromSmiles(header[0])
    if mol is not None:
        errors.add('First row is a SMILES string instead of a header.')

    # Validate smiles
    for smile in tqdm(smiles, total=len(smiles)):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            errors.add('Data includes an invalid SMILES.')

    # Validate targets
    num_tasks_set = set(len(mol_targets) for mol_targets in targets)
    if len(num_tasks_set) != 1:
        errors.add('Inconsistent number of tasks for each molecule.')

    if len(num_tasks_set) == 1:
        num_tasks = num_tasks_set.pop()
        if num_tasks != len(header) - 1:
            errors.add('Number of tasks for each molecule doesn\'t match number of tasks in header.')

    unique_targets = set(np.unique([target for mol_targets in targets for target in mol_targets]))

    if unique_targets <= {''}:
        errors.add('All targets are missing.')

    for target in unique_targets - {''}:
        try:
            float(target)
        except ValueError:
            errors.add('Found a target which is not a number.')

    return errors

from collections import defaultdict
from typing import Sequence

import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold


from chemprop.v2.data.datapoints import MoleculeDatapoint


def split_data(
    data: Sequence[MoleculeDatapoint],
    split: str = "random",
    sizes: tuple[float, float, float] = (0.8, 0.1, 0.1),
    k: int = 5,
    fold: int = 0,
):
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Invalid train/val/test splits! got: {sizes}")

    n_train, n_val, n_test = [int(size * len(data)) for size in sizes]

    if split == "random":
        idxs = np.arange(len(data))
        np.random.shuffle(idxs)

        train = [data[i] for i in idxs[:n_train]]
        val = [data[i] for i in idxs[n_train : n_train + n_val]]
        test = [data[i] for i in idxs[n_train + n_val :]]
    elif split == "scaffold":
        scaffold2idxs = defaultdict(set)
        for i, d in enumerate(data):
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=d.mol)
            scaffold2idxs[scaffold].add(i)

        big_index_sets = []
        small_index_sets = []
        for idxs in scaffold2idxs.values():
            if len(idxs) > n_val / 2 or len(idxs) > n_test / 2:
                big_index_sets.append(idxs)
            else:
                small_index_sets.append(idxs)

        np.random.shuffle(big_index_sets)
        np.random.shuffle(small_index_sets)
        idx_sets = [*big_index_sets, *small_index_sets]

        train_idxs, val_idxs, test_idxs = [], [], []
        for idxs in idx_sets:
            if len(train) + len(idxs) <= n_train:
                train_idxs.extend(idxs)
                train_scaffold_count += 1
            elif len(val) + len(idxs) <= n_val:
                val_idxs.extend(idxs)
                val_scaffold_count += 1
            else:
                test_idxs.extend(idxs)
                test_scaffold_count += 1

        train = [data[i] for i in train_idxs]
        val = [data[i] for i in val_idxs]
        test = [data[i] for i in test_idxs]
    else:
        raise ValueError(f"Uknown split type! got: {split}")

    return train, val, test
