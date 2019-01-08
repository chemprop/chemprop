from argparse import Namespace
from copy import deepcopy
from logging import Logger
import pickle
import random
from typing import List, Tuple
import os

import numpy as np
from tqdm import tqdm

from .data import MoleculeDatapoint, MoleculeDataset
from .scaffold import log_scaffold_stats, scaffold_split, scaffold_split_one, scaffold_split_overlap
from chemprop.features import load_features


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
             max_data_size: int = None,
             use_compound_names: bool = False) -> MoleculeDataset:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    :param path: Path to a CSV file.
    :param args: Arguments.
    :param max_data_size: The maximum number of data points to load.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A MoleculeDataset containing smiles strings and target values along
    with other info such as additional features and compound names when desired.
    """
    if args is not None:
        max_data_size = min(args.max_data_size or float('inf'), max_data_size or float('inf'))
        skip_smiles_path = args.skip_smiles_path

        if args.features_path:
            features_data = []
            for features_path in args.features_path:
                features_data.append(load_features(features_path))  # each is num_data x num_features
            features_data = np.concatenate(features_data, axis=1)
        else:
            features_data = None
    else:
        features_data = skip_smiles_path = None
        max_data_size = max_data_size or float('inf')

    # Load smiles to skip
    if skip_smiles_path is not None:
        with open(skip_smiles_path) as f:
            f.readline()  # skip header
            skip_smiles = {line.split(',')[0] for line in f}
    else:
        skip_smiles = set()

    # Load data
    with open(path) as f:
        f.readline()  # skip header
        lines = []
        for line in f:
            smiles = line.split(',')[0]

            if smiles == '':
                line = 'C' + line  # bandage for if your dataset has an empty line on rare occasions

            if smiles in skip_smiles:
                continue

            lines.append(line)

            if len(lines) >= max_data_size:
                break

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
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               args: Namespace = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    """
    Splits data into training, validation, and test splits.

    :param data: A MoleculeDataset.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :param args: Namespace of arguments.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if args is not None:
        maml, folds_file, val_fold_index, test_fold_index, scaffold_overlap = \
            args.maml, args.folds_file, args.val_fold_index, args.test_fold_index, args.scaffold_overlap
    else:
        folds_file = val_fold_index = test_fold_index = scaffold_overlap = None
        maml = False

    if maml:
        train_data, val_data, test_data = deepcopy(data), deepcopy(data), deepcopy(data)

        task_idxs = list(range(data.num_tasks()))
        random.seed(seed)
        random.shuffle(task_idxs)

        train_size = int(sizes[0] * data.num_tasks())
        train_val_size = int((sizes[0] + sizes[1]) * data.num_tasks())

        train_task_idxs = task_idxs[:train_size]
        val_task_idxs = task_idxs[train_size:train_val_size]
        test_task_idxs = task_idxs[train_val_size:]

        train_data.maml_init(train_task_idxs)
        val_data.maml_init(val_task_idxs)
        test_data.maml_init(test_task_idxs)

        return train_data, val_data, test_data

    if split_type == 'predetermined':
        assert sizes[2] == 0  # test set is created separately
        assert folds_file is not None
        assert test_fold_index is not None

        with open(folds_file, 'rb') as f:
            try:
                all_fold_indices = pickle.load(f)
            except UnicodeDecodeError:
                all_fold_indices = pickle.load(f, encoding='latin1')  # in case we're loading indices from python2
        assert len(data) == sum([len(fold_indices) for fold_indices in all_fold_indices])

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
            random.seed(seed)
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'scaffold':
        return scaffold_split(data, sizes=sizes, balanced=False, logger=logger)
    
    elif split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, seed=seed, logger=logger)

    elif split_type == 'scaffold_one':
        return scaffold_split_one(data)

    elif split_type == 'scaffold_overlap':
        assert scaffold_overlap is not None
        return scaffold_split_overlap(data, overlap=scaffold_overlap, seed=seed, logger=logger)

    elif split_type == 'random':
        data.shuffle(seed=seed)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError('split_type "{}" not supported.'.format(split_type))


def get_class_sizes(data: MoleculeDataset) -> List[List[float]]:
    """
    Determines the proportions of the different classes in the classification dataset.

    :param data: A classification dataset
    :return: A list of lists of class proportions. Each inner list contains the class proportions
    for a task.
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
        # Make sure we're dealing with a binary classification task
        assert set(np.unique(task_targets)) <= {0, 1}

        ones = np.count_nonzero(task_targets) / len(task_targets)
        class_sizes.append([1 - ones, ones])

    return class_sizes


def truncate_outliers(data: MoleculeDataset) -> MoleculeDataset:
    """Truncates outlier values in a regression dataset.

    Every value which is outside mean +/- 3 * std are truncated to equal mean +/- 3 * std.

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


def load_prespecified_chunks(args: Namespace, logger: Logger = None):
    """
    Load some number of chunks into train and val datasets. 

    :param args: Namespace of arguments.
    :param logger: An optional logger.
    :return: A tuple containing the train and validation MoleculeDatasets
    from loading a few random chunks. 
    """
    fnames = []
    for _, _, files in os.walk(args.prespecified_chunk_dir):
        fnames.extend(files)
    random.shuffle(fnames)

    data_len = 0
    chunks = []
    for fname in fnames:
        remaining_data_len = args.prespecified_chunks_max_examples_per_epoch - data_len
        path = os.path.join(args.prespecified_chunk_dir, fname)
        data = get_data(path, args, max_data_size=remaining_data_len)
        chunks.append(data)
        data_len += len(data)
        if data_len >= args.prespecified_chunks_max_examples_per_epoch:
            break

    data = [d for chunk in chunks for d in chunk.data]
    random.shuffle(data)
    data = MoleculeDataset(data)

    if args.dataset_type == 'bert_pretraining':
        data.bert_init(args, logger)

    split_sizes = deepcopy(args.split_sizes)
    split_sizes[2] = 0  # no test set
    split_sizes = [i / sum(split_sizes) for i in split_sizes]
    train, val, _ = split_data(data=data, split_type=args.split_type, sizes=split_sizes, args=args)

    return train, val

