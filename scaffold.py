from collections import defaultdict
from copy import deepcopy
import logging
from morgan_fingerprint import morgan_fingerprint
import random
from typing import Dict, List, Set, Tuple, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from data import MoleculeDataset


class ScaffoldGenerator:
    """
    Generate molecular scaffolds.
    Parameters
    ----------
    include_chirality : : bool, optional (default False)
        Include chirality in scaffolds.
    """

    def __init__(self, include_chirality: bool = False):
        self.include_chirality = include_chirality

    def get_scaffold(self, mol: Chem.rdchem.Mol) -> str:
        """
        Get Murcko scaffolds for molecules.
        Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
        essentially that part of the molecule consisting of rings and the
        linker atoms between them.
        Parameters
        ----------
        mol : array_like
            Molecules.
        """
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=self.include_chirality)


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    engine = ScaffoldGenerator(include_chirality=include_chirality)
    scaffold = engine.get_scaffold(mol)

    return scaffold


def scaffold_to_smiles(all_smiles: List[str], use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param all_smiles: A list of smiles strings.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, smiles in tqdm(enumerate(all_smiles), total=len(all_smiles)):
        scaffold = generate_scaffold(smiles)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(smiles)

    return scaffolds


def scaffold_split(data: MoleculeDataset,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   logger: logging.Logger = None) -> Tuple[MoleculeDataset,
                                                           MoleculeDataset,
                                                           MoleculeDataset]:
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data.smiles(), use_indices=True)

    # Sort from largest to smallest scaffold sets
    index_sets = sorted(list(scaffold_to_indices.values()),
                        key=lambda index_set: len(index_set),
                        reverse=True)

    # Split
    train_size, val_size = sizes[0] * len(data), sizes[1] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    if logger is not None:
        logger.debug('Total scaffolds = {:,} | train scaffolds = {:,} | val scaffolds = {:,} | test scaffolds = {:,}'.format(
            len(scaffold_to_indices),
            train_scaffold_count,
            val_scaffold_count,
            test_scaffold_count
        ))
    
    log_scaffold_stats(data, index_sets, logger=logger)

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


def log_scaffold_stats(data: MoleculeDataset,
                       index_sets: List[Set[int]],
                       num_scaffolds: int = 10,
                       num_labels: int = 20,
                       logger: logging.Logger = None) -> List[Tuple[List[float], List[int]]]:
    """
    Logs and returns statistics about counts and average target values in molecular scaffolds.

    :param data: A MoleculeDataset.
    :param index_sets: A list of sets of indices representing splits of the data.
    :param num_scaffolds: The number of scaffolds about which to display statistics.
    :param num_labels: The number of labels about which to display statistics.
    :param logger: A Logger.
    :return: A list of tuples where each tuple contains a list of average target values
    across the first num_labels labels and a list of the number of non-zero values for
    the first num_scaffolds scaffolds, sorted in decreasing order of scaffold frequency.
    """
    # print some statistics about scaffolds
    target_avgs = []
    counts = []
    for index_set in index_sets:
        data_set = [data[i] for i in index_set]
        targets = [d.targets for d in data_set]
        targets = np.array(targets, dtype=np.float)
        target_avgs.append(np.nanmean(targets, axis=0))
        counts.append(np.count_nonzero(~np.isnan(targets), axis=0))
    stats = [(target_avgs[i][:num_labels], counts[i][:num_labels]) for i in range(min(num_scaffolds, len(target_avgs)))]

    if logger is not None:
        logger.debug('Label averages per scaffold, in decreasing order of scaffold frequency,'
                     'capped at {} scaffolds and {} labels: {}'.format(num_scaffolds, num_labels, stats))

    return stats


def scaffold_split_one(data: MoleculeDataset) -> Tuple[MoleculeDataset,
                                                       MoleculeDataset,
                                                       MoleculeDataset]:
    """
    Split a dataset by scaffold such that train has all molecules from the largest scaffold
    (i.e. the scaffold with the most molecules), val has all molecules from the second largest
    scaffold, and test has all molecules from the third largest scaffold.

    :param data: A MoleculeDataset.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data.smiles(), use_indices=True)

    # Sort from largest to smallest scaffold sets
    scaffolds = sorted(list(scaffold_to_indices.keys()),
                       key=lambda scaffold: len(scaffold_to_indices[scaffold]),
                       reverse=True)

    train = [data[index] for index in scaffold_to_indices[scaffolds[0]]]
    val = [data[index] for index in scaffold_to_indices[scaffolds[1]]]
    test = [data[index] for index in scaffold_to_indices[scaffolds[2]]]

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


def cluster_split(data: MoleculeDataset,
                  n_clusters: int,
                  seed: int = 0,
                  logger: logging.Logger = None) -> List[MoleculeDataset]:
    """
    Split a dataset by K-means clustering on Morgan fingerprints. 

    :param data: A list of data points (smiles string, target values).
    :param n_clusters: Number of clusters for K-means. 
    :param seed: Random seed for K-means. 
    :param logger: A logger for logging cluster split stats.
    :return: A list containing the K-means splits.
    """
    fp = [morgan_fingerprint(s) for s in data.smiles()]
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)
    cluster_labels = kmeans.fit_predict(fp)

    clusters = [[] for _ in range(n_clusters)]
    for i in range(len(data)):
        clusters[cluster_labels[i]].append(data[i])
    
    if logger is not None:
        logger.debug('Split into {} clusters'.format(n_clusters))
        logger.debug('Cluster sizes: {}'.format([len(c) for c in clusters]))

    return [MoleculeDataset(cluster) for cluster in clusters]


def scaffold_overlap(indices_1: Set[int],
                     indices_2: Set[int],
                     index_to_scaffold: Dict[int, str]) -> float:
    """
    Computes the proportion of indices in indices_2 which have a scaffold in indices_1.

    :param indices_1: A set of indices (that originally map to smiles strings).
    :param indices_2: A set of indices (that originally map to smiles strings).
    :param index_to_scaffold: A dictionary mapping index to scaffold string.
    :return: The proportion of indices in indices_2 which have a scaffold in indices_1.
    """
    scaffolds_1 = {index_to_scaffold[index] for index in indices_1}
    indices_in_2_with_scaffold_in_1 = {index for index in indices_2 if index_to_scaffold[index] in scaffolds_1}
    overlap = len(indices_in_2_with_scaffold_in_1) / len(indices_2)

    return overlap


def decrease_overlap(indices_1: Set[int],
                     indices_2: Set[int],
                     index_to_scaffold: Dict[int, str],
                     scaffold_to_indices: Dict[str, Set[int]],
                     size_ratio: float) -> Tuple[Set[int], Set[int]]:
    """
    Decrease the scaffold overlap between two sets of indices by selecting two shared
    scaffolds and moving all indices to either the first or second set according to size.

    Specifically, this algorithm works by selecting a random shared scaffold and
    moving all indices in it to indices_2 and then selecting the shared scaffold with
    size closest to size_ratio * (size of that first shared scaffold) and moves all of
    those indices to indices_1.

    :param indices_1: A set of indices (that originally map to smiles strings).
    :param indices_2: A set of indices (that originally map to smiles strings).
    :param index_to_scaffold: A dictionary mapping index to scaffold string.
    :param scaffold_to_indices: A dictionary mapping scaffold string to a set of indices.
    :param size_ratio: Desired ratio len(indices_1)/len(indices_2).
    :return: The two sets of indices with decreased overlap.
    """
    # Make copies to prevent altering input set
    indices_1 = deepcopy(indices_1)
    indices_2 = deepcopy(indices_2)

    # Determine scaffolds in each of the two sets
    scaffolds_1 = {index_to_scaffold[index] for index in indices_1}
    scaffolds_2 = {index_to_scaffold[index] for index in indices_2}
    union = scaffolds_1 | scaffolds_2
    intersection = scaffolds_1 & scaffolds_2

    # Return indices in cases when overlap can't be changed
    # TODO: deal with intersection of size 1
    if len(union) <= 1 or len(intersection) <= 1:
        return indices_1, indices_2

    # Select random scaffold and move all indices to indices_2
    scaffold = random.choice(list(intersection))
    indices = scaffold_to_indices[scaffold]
    indices_1 -= indices
    indices_2 |= indices
    intersection.remove(scaffold)

    # Select scaffold which is closest in size to above scaffold
    first_scaffold_size = len(indices)
    best_size_diff = float('inf')
    best_scaffold = None

    for scaffold in intersection:
        second_scaffold_size = len(scaffold_to_indices[scaffold])
        size_diff = abs(first_scaffold_size / second_scaffold_size - size_ratio)

        if size_diff < best_size_diff:
            best_size_diff = size_diff
            best_scaffold = scaffold

    # Move all indices of this scaffold to indices_1
    indices = scaffold_to_indices[best_scaffold]
    indices_2 -= indices
    indices_1 |= indices

    return indices_1, indices_2


def increase_overlap(indices_1: Set[int],
                     indices_2: Set[int],
                     index_to_scaffold: Dict[int, str],
                     scaffold_to_indices: Dict[str, Set[int]],
                     size_ratio: float) -> Tuple[Set[int], Set[int]]:
    """
    Increase the scaffold overlap between two sets of indices by randomly selecting two unshared
    scaffolds, one in each set, and splitting the indices evenly between the two sets.

    :param indices_1: A set of indices (that originally map to smiles strings).
    :param indices_2: A set of indices (that originally map to smiles strings).
    :param index_to_scaffold: A dictionary mapping index to scaffold string.
    :param scaffold_to_indices: A dictionary mapping scaffold string to a set of indices.
    :param size_ratio: Desired ratio len(indices_1)/len(indices_2).
    :return: The two sets of indices with increased overlap.
    """
    # Make copies to prevent altering input set
    indices_1 = deepcopy(indices_1)
    indices_2 = deepcopy(indices_2)

    # Determine scaffolds in each of the two sets
    scaffolds_1 = {index_to_scaffold[index] for index in indices_1}
    scaffolds_2 = {index_to_scaffold[index] for index in indices_2}
    union = scaffolds_1 | scaffolds_2

    # If 0 or 1 scaffolds, can't increase overlap so just return indices
    if len(union) <= 1:
        return indices_1, indices_2

    # Determine scaffolds which are only in one set or the other
    scaffolds_1_only = scaffolds_1 - scaffolds_2
    scaffolds_2_only = scaffolds_2 - scaffolds_1

    # Select one scaffold from each set if possible
    selected_scaffolds = []

    if len(scaffolds_1_only) != 0:
        selected_scaffolds.append(random.choice(list(scaffolds_1_only)))
    if len(scaffolds_2_only) != 0:
        selected_scaffolds.append(random.choice(list(scaffolds_2_only)))

    # Share indices from selected scaffolds
    for scaffold in selected_scaffolds:
        indices = scaffold_to_indices[scaffold]

        indices_1 -= indices
        indices_2 -= indices

        indices = list(indices)
        random.shuffle(indices)

        # Divide up indices proportionally according to size_ratio
        size_1 = int(size_ratio * len(indices))
        indices_1.update(indices[:size_1])
        indices_2.update(indices[size_1:])

    return indices_1, indices_2


def scaffold_split_overlap(data: MoleculeDataset,
                           overlap: float,
                           overlap_error: float = 0.05,
                           max_attempts: int = 200,
                           sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                           seed: int = 0,
                           logger: logging.Logger = None) -> Tuple[MoleculeDataset,
                                                                   MoleculeDataset,
                                                                   MoleculeDataset]:
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A MoleculeDataset.
    :param overlap: The proportion of test molecules which should share a molecular
    scaffold with at least one train molecule.
    :param overlap_error: The  amount by which the overlap proportion generated
    is allowed to deviate from the overlap requested.
    :param max_attempts: Maximum number of attempts to achieve an overlap within overlap_error
    of the desired overlap before giving up and crashing.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: Random seed.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3 and sum(sizes) == 1 and 0 <= overlap <= 1 and overlap_error >= 0

    # Random seed
    random.seed(seed)  # TODO: randomness is not consistent

    # Start with random split of the data
    train_size = int(sizes[0] * len(data))
    train, val_test = set(range(train_size)), set(range(train_size, len(data)))

    # Map from scaffold to index and index to scaffold
    scaffold_to_indices = scaffold_to_smiles(data.smiles(), use_indices=True)
    index_to_scaffold = {index: scaffold for scaffold, index_set in scaffold_to_indices.items() for index in index_set}

    # Get overlap bounds
    min_overlap, max_overlap = overlap - overlap_error, overlap + overlap_error

    # Get size ratio
    size_ratio = sizes[0] / (sizes[1] + sizes[2])

    # Adjust train/val/test sets to achieve desired overlap
    for attempt in range(max_attempts + 1):
        # Get current overlap
        current_overlap = scaffold_overlap(train, val_test, index_to_scaffold)

        print('train size', len(train) / len(data))
        print('val_test size', len(val_test) / len(data))
        print()
        print('current overlap', current_overlap)
        print()

        # If overlap is within error bounds, break
        if min_overlap <= current_overlap <= max_overlap:
            print(attempt)
            break

        # If reached max attempts unsuccessfully, raise error
        if attempt == max_attempts:
            raise Exception('Unable to achieve desired scaffold overlap after {} attempts :('.format(attempt))

        # Adjust overlap balance
        if current_overlap > max_overlap:
            train, val_test = decrease_overlap(train, val_test, index_to_scaffold, scaffold_to_indices, size_ratio)
        else:
            train, val_test = increase_overlap(train, val_test, index_to_scaffold, scaffold_to_indices, size_ratio)

    # Split val/test
    train_scaffolds = {index_to_scaffold[index] for index in train}

    val_test_overlap = [index for index in val_test if index_to_scaffold[index] in train_scaffolds]
    val_test_non_overlap = [index for index in val_test if index_to_scaffold[index] not in train_scaffolds]

    val_test_ratio = sizes[1] / (sizes[1] + sizes[2])

    val_overlap_size = int(val_test_ratio * len(val_test_overlap))
    val_non_overlap_size = int(val_test_ratio * len(val_test_non_overlap))

    val = val_test_overlap[:val_overlap_size] + val_test_non_overlap[:val_non_overlap_size]
    test = val_test_overlap[val_overlap_size:] + val_test_non_overlap[val_non_overlap_size:]



    # temp
    val, test = set(val), set(test)

    print('train size', len(train) / len(data))
    print('val size', len(val) / len(data))
    print('test size', len(test) / len(data))
    print()
    print('val overlap', scaffold_overlap(train, val, index_to_scaffold))
    print('test overlap', scaffold_overlap(train, test, index_to_scaffold))

    exit()


    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    # Shuffle since overlap and non-overlap are not shuffled
    random.seed(seed)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)
