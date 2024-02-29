from collections import OrderedDict, defaultdict
import sys
import csv
import ctypes
from logging import Logger
import pickle
from random import Random
from typing import List, Set, Tuple, Union
import os
import json

from rdkit import Chem
import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import MoleculeDatapoint, MoleculeDataset, make_mols
from .scaffold import log_scaffold_stats, scaffold_split
from chemprop.args import PredictArgs, TrainArgs
from chemprop.features import load_features, load_valid_atom_or_bond_features, is_mol
from chemprop.rdkit import make_mol

# Increase maximum size of field in the csv processing for the current architecture
csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

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
        if isinstance(smiles_columns, str):
            smiles_columns = [smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError('Length of smiles_columns must match number_of_molecules.')
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError('Provided smiles_columns do not match the header of data file.')

    return smiles_columns


def get_task_names(
    path: str,
    smiles_columns: Union[str, List[str]] = None,
    target_columns: List[str] = None,
    ignore_columns: List[str] = None,
    loss_function: str = None,
) -> List[str]:
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

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    ignore_columns = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))

    target_names = [column for column in columns if column not in ignore_columns]

    if loss_function == "quantile_interval":
        target_names = target_names * 2

    return target_names


def get_mixed_task_names(path: str,
                         smiles_columns: Union[str, List[str]] = None,
                         target_columns: List[str] = None,
                         ignore_columns: List[str] = None,
                         keep_h: bool = None,
                         add_h: bool = None,
                         keep_atom_map: bool = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Gets the task names for atomic, bond, and molecule targets separately from a data CSV file.

    If :code:`target_columns` is provided, returned lists based off `target_columns`.
    Otherwise, returned lists based off all columns except the :code:`smiles_columns`
    (or the first column, if the :code:`smiles_columns` is None) and
    the :code:`ignore_columns`.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_columns` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h: Boolean whether to add hydrogens to the input smiles.
    :param keep_atom_map: Boolean whether to keep the original atom mapping.
    :return: A tuple containing the task names of atomic, bond, and molecule properties separately.
    """
    columns = get_header(path)

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    ignore_columns = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))

    if target_columns is not None:
        target_names =  target_columns
    else:
        target_names = [column for column in columns if column not in ignore_columns]

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            atom_target_names, bond_target_names, molecule_target_names = [], [], []
            smiles = [row[c] for c in smiles_columns]
            for s in smiles:
                if keep_atom_map:
                    # When the original atom mapping is used, the explicit hydrogens specified in the input SMILES should be used
                    # However, the explicit Hs can only be added for reactions with `--explicit_h` flag
                    # To fix this, `keep_h` is set to True when `keep_atom_map` is also True
                    mol = make_mol(s, keep_h=True, add_h=add_h, keep_atom_map=True)
                else:
                    mol = make_mol(s, keep_h=keep_h, add_h=add_h, keep_atom_map=False)
                if len(mol.GetAtoms()) != len(mol.GetBonds()):
                    break

            for column in target_names:
                value = row[column]
                value = value.replace('None', 'null')
                target = np.array(json.loads(value))

                is_atom_target, is_bond_target, is_molecule_target = False, False, False
                if len(target.shape) == 0:
                    is_molecule_target = True
                elif len(target.shape) == 1:
                    if len(target) == len(mol.GetAtoms()):  # Atom targets saved as 1D list
                        is_atom_target = True
                    elif len(target) == len(mol.GetBonds()):  # Bond targets saved as 1D list
                        is_bond_target = True
                    else:
                        raise RuntimeError(f'Unrecognized targets of column {column} in {path}. '
                                           'Expected targets should be either atomic or bond targets. '
                                           'Please ensure the content is correct.')
                elif len(target.shape) == 2:  # Bond targets saved as 2D list
                    is_bond_target = True
                else:
                    raise ValueError(f'Unrecognized targets of column {column} in {path}.')
                
                if is_atom_target:
                    atom_target_names.append(column)
                elif is_bond_target:
                    bond_target_names.append(column)
                elif is_molecule_target:
                    molecule_target_names.append(column)
            if len(atom_target_names) + len(bond_target_names) + len(molecule_target_names) == len(target_names):
                break

    return atom_target_names, bond_target_names, molecule_target_names


def get_data_weights(path: str) -> List[float]:
    """
    Returns the list of data weights for the loss function as stored in a CSV file.

    :param path: Path to a CSV file.
    :return: A list of floats containing the data weights.
    """
    weights = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for line in reader:
            weights.append(float(line[0]))
    # normalize the data weights
    avg_weight = sum(weights) / len(weights)
    weights = [w / avg_weight for w in weights]
    if min(weights) < 0:
        raise ValueError('Data weights must be non-negative for each datapoint.')
    return weights


def get_constraints(path: str,
                    target_columns: List[str],
                    save_raw_data: bool = False) -> Tuple[List[float], List[float]]:
    """
    Returns lists of data constraints for the atomic/bond targets as stored in a CSV file.

    :param path: Path to a CSV file.
    :param target_columns: Name of the columns containing target values.
    :param save_raw_data: Whether to save all user-provided atom/bond-level constraints in input data,
                          which will be used to construct constraints files for each train/val/test split
                          for prediction convenience later.
    :return: Lists of floats containing the data constraints.
    """
    constraints_data = []
    reader = pd.read_csv(path)
    reader_columns = reader.columns.tolist()
    if len(reader_columns) != len(set(reader_columns)):
        raise ValueError(f'There are duplicates in {path}.')
    for target in target_columns:
        if target in reader_columns:
            constraints_data.append(reader[target].values)
        else:
            constraints_data.append([None] * len(reader))
    constraints_data = np.transpose(constraints_data)  # each is num_data x num_targets

    if save_raw_data:
        raw_constraints_data = []
        for target in reader_columns:
            raw_constraints_data.append(reader[target].values)
        raw_constraints_data = np.transpose(raw_constraints_data)  # each is num_data x num_columns
    else:
        raw_constraints_data = None
    
    return constraints_data, raw_constraints_data


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

    if (isinstance(smiles_columns, str) or smiles_columns is None) and header:
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
    keep_atom_map_list = [False for x in is_mol_list]  # set this to False as it is not needed for invalid SMILES check

    for mol_smiles in smiles:
        mols = make_mols(smiles=mol_smiles, reaction_list=is_reaction_list, keep_h_list=is_explicit_h_list,
                         add_h_list=is_adding_hs_list, keep_atom_map_list=keep_atom_map_list)
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
             args: Union[TrainArgs, PredictArgs] = None,
             data_weights_path: str = None,
             features_path: List[str] = None,
             features_generator: List[str] = None,
             phase_features_path: str = None,
             atom_descriptors_path: str = None,
             bond_descriptors_path: str = None,
             constraints_path: str = None,
             max_data_size: int = None,
             store_row: bool = False,
             logger: Logger = None,
             loss_function: str = None,
             skip_none_targets: bool = False) -> MoleculeDataset:
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
    :param bond_descriptors_path: The path to the file containing the custom bond descriptors.
    :param constraints_path: The path to the file containing constraints applied to different atomic/bond properties.
    :param max_data_size: The maximum number of data points to load.
    :param logger: A logger for recording output.
    :param store_row: Whether to store the raw CSV row in each :class:`~chemprop.data.data.MoleculeDatapoint`.
    :param skip_none_targets: Whether to skip targets that are all 'None'. This is mostly relevant when --target_columns
                              are passed in, so only a subset of tasks are examined.
    :param loss_function: The loss function to be used in training.
    :return: A :class:`~chemprop.data.MoleculeDataset` containing SMILES and target values along
             with other info such as additional features when desired.
    """
    debug = logger.debug if logger is not None else print

    if args is not None:
        # Prefer explicit function arguments but default to args if not provided
        smiles_columns = smiles_columns if smiles_columns is not None else args.smiles_columns
        target_columns = target_columns if target_columns is not None else args.target_columns
        ignore_columns = ignore_columns if ignore_columns is not None else args.ignore_columns
        features_path = features_path if features_path is not None else args.features_path
        features_generator = features_generator if features_generator is not None else args.features_generator
        phase_features_path = phase_features_path if phase_features_path is not None else args.phase_features_path
        atom_descriptors_path = atom_descriptors_path if atom_descriptors_path is not None \
            else args.atom_descriptors_path
        bond_descriptors_path = bond_descriptors_path if bond_descriptors_path is not None \
            else args.bond_descriptors_path
        constraints_path = constraints_path if constraints_path is not None else args.constraints_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        loss_function = loss_function if loss_function is not None else args.loss_function

    if isinstance(smiles_columns, str) or smiles_columns is None:
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
            features_data = np.concatenate((features_data, phase_features), axis=1)
        else:  # if there are no other molecular features, phase features become the only molecular features
            features_data = np.array(phase_features)
    else:
        phase_features = None

    # Load constraints
    if constraints_path is not None:
        constraints_data, raw_constraints_data = get_constraints(
            path=constraints_path,
            target_columns=args.target_columns,
            save_raw_data=args.save_smiles_splits
        )
    else:
        constraints_data = None
        raw_constraints_data = None

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
            loss_function=loss_function,
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

        all_smiles, all_targets, all_atom_targets, all_bond_targets, all_rows, all_features, all_phase_features, all_constraints_data, all_raw_constraints_data, all_weights, all_gt, all_lt = [], [], [], [], [], [], [], [], [], [], [], []
        for i, row in enumerate(tqdm(reader)):
            smiles = [row[c] for c in smiles_columns]

            targets, atom_targets, bond_targets = [], [], []
            for column in target_columns:
                value = row[column]
                if value in ['', 'nan']:
                    targets.append(None)
                elif '>' in value or '<' in value:
                    if loss_function == 'bounded_mse':
                        targets.append(float(value.strip('<>')))
                    else:
                        raise ValueError('Inequality found in target data. To use inequality targets (> or <), the regression loss function bounded_mse must be used.')
                elif '[' in value or ']' in value:
                    value = value.replace('None', 'null')
                    target = np.array(json.loads(value))
                    if len(target.shape) == 1 and column in args.atom_targets:  # Atom targets saved as 1D list
                        atom_targets.append(target)
                        targets.append(target)
                    elif len(target.shape) == 1 and column in args.bond_targets:  # Bond targets saved as 1D list
                        bond_targets.append(target)
                        targets.append(target)
                    elif len(target.shape) == 2:  # Bond targets saved as 2D list
                        bond_target_arranged = []
                        mol = make_mol(smiles[0], args.explicit_h, args.adding_h, args.keeping_atom_map)
                        for bond in mol.GetBonds():
                            bond_target_arranged.append(target[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                        bond_targets.append(np.array(bond_target_arranged))
                        targets.append(np.array(bond_target_arranged))
                    else:
                        raise ValueError(f'Unrecognized targets of column {column} in {path}.')
                else:
                    targets.append(float(value))

            # Check whether all targets are None and skip if so
            if skip_none_targets and all(x is None for x in targets):
                continue

            all_smiles.append(smiles)
            all_targets.append(targets)
            all_atom_targets.append(atom_targets)
            all_bond_targets.append(bond_targets)

            if features_data is not None:
                all_features.append(features_data[i])

            if phase_features is not None:
                all_phase_features.append(phase_features[i])

            if constraints_data is not None:
                all_constraints_data.append(constraints_data[i])

            if raw_constraints_data is not None:
                all_raw_constraints_data.append(raw_constraints_data[i])

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
        if args is not None and args.atom_descriptors is not None:
            try:
                descriptors = load_valid_atom_or_bond_features(atom_descriptors_path, [x[0] for x in all_smiles])
            except Exception as e:
                raise ValueError(f'Failed to load or validate custom atomic descriptors or features: {e}')

            if args.atom_descriptors == 'feature':
                atom_features = descriptors
            elif args.atom_descriptors == 'descriptor':
                atom_descriptors = descriptors

        bond_features = None
        bond_descriptors = None
        if args is not None and args.bond_descriptors is not None:
            try:
                descriptors = load_valid_atom_or_bond_features(bond_descriptors_path, [x[0] for x in all_smiles])
            except Exception as e:
                raise ValueError(f'Failed to load or validate custom bond descriptors or features: {e}')

            if args.bond_descriptors == 'feature':
                bond_features = descriptors
            elif args.bond_descriptors == 'descriptor':
                bond_descriptors = descriptors

        data = MoleculeDataset([
            MoleculeDatapoint(
                smiles=smiles,
                targets=targets,
                atom_targets=all_atom_targets[i] if atom_targets else None,
                bond_targets=all_bond_targets[i] if bond_targets else None,
                row=all_rows[i] if store_row else None,
                data_weight=all_weights[i] if data_weights is not None else None,
                gt_targets=all_gt[i] if gt_targets is not None else None,
                lt_targets=all_lt[i] if lt_targets is not None else None,
                features_generator=features_generator,
                features=all_features[i] if features_data is not None else None,
                phase_features=all_phase_features[i] if phase_features is not None else None,
                atom_features=atom_features[i] if atom_features is not None else None,
                atom_descriptors=atom_descriptors[i] if atom_descriptors is not None else None,
                bond_features=bond_features[i] if bond_features is not None else None,
                bond_descriptors=bond_descriptors[i] if bond_descriptors is not None else None,
                constraints=all_constraints_data[i] if constraints_data is not None else None,
                raw_constraints=all_raw_constraints_data[i] if raw_constraints_data is not None else None,
                overwrite_default_atom_features=args.overwrite_default_atom_features if args is not None else False,
                overwrite_default_bond_features=args.overwrite_default_bond_features if args is not None else False
            ) for i, (smiles, targets) in tqdm(enumerate(zip(all_smiles, all_targets)),
                                            total=len(all_smiles))
        ])

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

def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               key_molecule_index: int = 0,
               seed: int = 0,
               num_folds: int = 1,
               args: TrainArgs = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
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
        raise ValueError(f"Split sizes do not sum to 1. Received train/val/test splits: {sizes}")
    if any([size < 0 for size in sizes]):
        raise ValueError(f"Split sizes must be non-negative. Received train/val/test splits: {sizes}")

    random = Random(seed)

    if args is not None:
        folds_file, val_fold_index, test_fold_index = \
            args.folds_file, args.val_fold_index, args.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None

    if split_type == 'crossval':
        index_set = args.crossval_index_sets[args.seed]
        data_split = []
        for split in range(3):
            split_indices = []
            for index in index_set[split]:
                with open(os.path.join(args.crossval_index_dir, f'{index}.pkl'), 'rb') as rf:
                    split_indices.extend(pickle.load(rf))
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type in {'cv', 'cv-no-test'}:
        if num_folds <= 1 or num_folds > len(data):
            raise ValueError(f'Number of folds for cross-validation must be between 2 and the number of valid datapoints ({len(data)}), inclusive.')

        random = Random(0)

        indices = np.tile(np.arange(num_folds), 1 + len(data) // num_folds)[:len(data)]
        random.shuffle(indices)
        test_index = seed % num_folds
        val_index = (seed + 1) % num_folds

        train, val, test = [], [], []
        for d, index in zip(data, indices):
            if index == test_index and split_type != 'cv-no-test':
                test.append(d)
            elif index == val_index:
                val.append(d)
            else:
                train.append(d)

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'index_predetermined':
        split_indices = args.crossval_index_sets[args.seed]

        if len(split_indices) != 3:
            raise ValueError('Split indices must have three splits: train, validation, and test')

        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'predetermined':
        if not val_fold_index and sizes[2] != 0:
            raise ValueError('Test size must be zero since test set is created separately '
                             'and we want to put all other data in train and validation')

        if folds_file is None:
            raise ValueError('arg "folds_file" can not be None!')
        if test_fold_index is None:
            raise ValueError('arg "test_fold_index" can not be None!')

        try:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f, encoding='latin1')  # in case we're loading indices from python2

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

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, key_molecule_index=key_molecule_index, seed=seed, logger=logger)

    elif split_type == 'random_with_repeated_smiles':  # Use to constrain data with the same smiles go in the same split.
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
            if len(train)+len(index_set) <= train_size:
                train += index_set
            elif len(val) + len(index_set) <= val_size:
                val += index_set
            else:
                test += index_set
        train = [data[i] for i in train]
        val = [data[i] for i in val]
        test = [data[i] for i in test]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'random':
        indices = list(range(len(data)))
        random.shuffle(indices)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = [data[i] for i in indices[:train_size]]
        val = [data[i] for i in indices[train_size:train_val_size]]
        test = [data[i] for i in indices[train_val_size:]]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)
    elif split_type == 'molecular_weight':
        train_size, val_size, test_size = [int(size * len(data)) for size in sizes]

        sorted_data = sorted(data._data, key=lambda x: x.max_molwt, reverse=False)
        indices = list(range(len(sorted_data)))

        train_end_idx = int(train_size)
        val_end_idx = int(train_size + val_size)
        train_indices = indices[:train_end_idx]
        val_indices = indices[train_end_idx:val_end_idx]
        test_indices = indices[val_end_idx:]

        # Create MoleculeDataset for each split
        train = MoleculeDataset([sorted_data[i] for i in train_indices])
        val = MoleculeDataset([sorted_data[i] for i in val_indices])
        test = MoleculeDataset([sorted_data[i] for i in test_indices])

        return train, val, test
    else:
        raise ValueError(f'split_type "{split_type}" not supported.')


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
            if data.is_atom_bond_targets:
                for target in targets[i][task_num]:
                    if targets[i][task_num] is not None:
                        valid_targets[task_num].append(target)
            else:
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
        else:  # counts
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
    target_list = [target for targets in data.targets() for target in targets]

    if data.is_atom_bond_targets:
        target_set = set(list(np.concatenate(target_list).flat)) - {None}
    else:
        target_set = set(target_list) - {None}
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
