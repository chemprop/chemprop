from argparse import Namespace
from copy import deepcopy
import logging
import pickle
import random
from typing import List, Tuple
import os

import numpy as np
from tqdm import tqdm

from .data import MoleculeDatapoint, MoleculeDataset
from .scaffold import log_scaffold_stats, scaffold_split, scaffold_split_one, scaffold_split_overlap
from chemprop.features import get_features


def convert_to_classes(data: MoleculeDataset, num_bins: int = 20) -> Tuple[MoleculeDataset,
                                                                           np.ndarray,
                                                                           MoleculeDataset]:
    """
    Converts regression data to classification data by binning.

    :param data: Regression data as a list of molecule datapoints.
    :param num_bins: The number of bins to use when doing regression_with_binning.
    :return: A tuple with the new classification data, a numpy array with the bin centers,
    and the original regression data.
    """
    print('Num bins for binning: {}'.format(num_bins))
    old_data = deepcopy(data)
    for task in range(data.num_tasks()):
        regress = np.array([targets[task] for targets in data.targets()])
        bin_edges = np.quantile(regress, [float(i) / float(num_bins) for i in range(num_bins + 1)])

        for i in range(len(data)):
            bin_index = (bin_edges <= regress[i]).sum() - 1
            bin_index = min(bin_index, num_bins - 1)
            data[i].targets[task] = bin_index

    return data, np.array([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(num_bins)]), old_data


def get_task_names(path: str, use_compound_names: bool = False) -> List[str]:
    """
    Gets the task names from a data CSV file.

    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A list of task names.
    """
    index = 2 if use_compound_names else 1
    with open(path) as f:
        task_names = f.readline().strip().split(',')[index:]

    return task_names


def get_desired_labels(args: Namespace, task_names: List[str]) -> List[str]:
    if args.show_individual_scores and args.labels_to_show:
        desired_labels = []
        with open(args.labels_to_show, 'r') as f:
            for line in f:
                desired_labels.append(line.strip())
    else:
        desired_labels = task_names
    return desired_labels


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = f.readline().strip().split(',')

    return header


def get_data(path: str,
             args: Namespace = None,
             use_compound_names: bool = False) -> MoleculeDataset:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    :param path: Path to a CSV file.
    :param args: Arguments.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A MoleculeDataset containing smiles strings and target values along
    with other info such as additional features and compound names when desired.
    """
    if args is not None and args.features_path:
        features_data = get_features(args.features_path)
    else:
        features_data = None

    with open(path) as f:
        f.readline()  # skip header
        lines = f.readlines()
        data = MoleculeDataset([
            MoleculeDatapoint(
                line=line.strip().split(','),
                args=args,
                features=features_data[i] if features_data is not None else None,
                use_compound_names=use_compound_names,
            ) for i, line in tqdm(enumerate(lines), total=len(lines))])

    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)

    if args is not None and args.dataset_type == 'regression_with_binning':
        data = convert_to_classes(data, args.num_bins)

    return data


def split_data(data: MoleculeDataset,
               args: Namespace,
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               logger: logging.Logger = None) -> Tuple[MoleculeDataset,
                                                       MoleculeDataset,
                                                       MoleculeDataset]:
    """
    Splits data into training, validation, and test splits.

    :param data: A MoleculeDataset.
    :param args: Namespace of arguments
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3, sum(sizes) == 1

    if args.split_type == 'predetermined':
        assert sizes[2] == 0  # test set is created separately
        with open(args.folds_file, 'rb') as f:
            all_fold_indices = pickle.load(f)
        assert len(data) == sum([len(fold_indices) for fold_indices in all_fold_indices])

        log_scaffold_stats(data, all_fold_indices, logger=logger)

        folds = [[data[i] for i in fold_indices] for fold_indices in all_fold_indices]

        test = folds[args.test_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != args.test_fold_index:
                train_val.extend(folds[i])

        random.seed(seed)
        random.shuffle(train_val)
        train_size = int(sizes[0] * len(train_val))
        train = train_val[:train_size]
        val = train_val[train_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif args.split_type == 'scaffold':
        return scaffold_split(data, sizes=sizes, logger=logger)

    elif args.split_type == 'scaffold_one':
        return scaffold_split_one(data)

    elif args.split_type == 'scaffold_overlap':
        return scaffold_split_overlap(data, overlap=args.scaffold_overlap, seed=seed, logger=logger)

    elif args.split_type == 'random':
        data.shuffle(seed=seed)

        train_size, val_size = [int(size * len(data)) for size in sizes[:2]]

        train = data[:train_size]
        val = data[train_size:train_size + val_size]
        test = data[train_size + val_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError('split_type "{}" not supported.'.format(args.split_type))


def truncate_outliers(data: MoleculeDataset) -> MoleculeDataset:
    """Truncates outlier values in a regression dataset.

    Every value which is outside mean ± 3 * std are truncated to equal mean ± 3 * std.

    :param data: A MoleculeDataset.
    :return: The same data but with outliers truncated.
    """
    # Determine mean and standard deviation by task
    smiles, targets = data.smiles(), data.targets()
    targets_by_task = np.array(targets).T
    means = np.mean(targets, axis=0)
    stds = np.std(targets, axis=0)

    # Truncate values
    for i, task_values in enumerate(targets_by_task):
        targets_by_task[i] = np.clip(task_values, means[i] - 3 * stds[i], means[i] + 3 * stds[i])

    # Reconstruct data
    targets = targets_by_task.T.tolist()
    for i in range(len(data)):
        data[i].targets = targets[i]

    return data

def load_prespecified_chunks(args: Namespace):
    """
    Load some number of chunks into train and val datasets. 

    :param args: Namespace of arguments
    :return: A tuple containing the train and validation MoleculeDatasets
    from loading a few random chunks. 
    """
    chunks = []
    for _, _, names in os.walk(args.prespecified_chunk_dir):
        random.shuffle(names)
    data_len = 0
    for name in names:
        path = os.path.join(args.prespecified_chunk_dir, name)
        chunks.append(get_data(path, args))
        data_len += len(chunks[-1].data)
        if data_len > args.prespecified_chunks_max_examples_per_epoch:
            break
    data = [c.data for c in chunks]
    full_data = []
    for d in data:
        full_data += d
    random.shuffle(full_data)
    full_data = full_data[:args.prespecified_chunks_max_examples_per_epoch]
    full_data = MoleculeDataset(full_data)
    split_sizes = deepcopy(args.split_sizes)
    split_sizes[2] = 0 # no test set
    split_sizes = [i / sum(split_sizes) for i in split_sizes]
    train, val, _ = split_data(full_data, args, split_sizes)
    return train, val

