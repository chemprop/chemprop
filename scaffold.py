from collections import defaultdict
import logging
from typing import Dict, List, Set, Tuple, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import numpy as np

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


def scaffold_split(data: List[Tuple[str, List[float]]],
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   logger: logging.Logger = None) -> Tuple[List[Tuple[str, List[float]]],
                                                           List[Tuple[str, List[float]]],
                                                           List[Tuple[str, List[float]]]]:
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A list of data points (smiles string, target values).
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    # Get data
    smiles, _ = zip(*data)
    scaffold_to_indices_map = scaffold_to_smiles(smiles, use_indices=True)

    # Sort from largest to smallest scaffold sets
    index_sets = sorted(list(scaffold_to_indices_map.values()),
                        key=lambda index_set: len(index_set),
                        reverse=True)

    # Split
    train_size, val_size = sizes[0] * len(data), sizes[1] * len(data)
    train_indices, val_indices, test_indices = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    for index_set in index_sets:
        if len(train_indices) + len(index_set) <= train_size:
            train_indices += index_set
            train_scaffold_count += 1
        elif len(val_indices) + len(index_set) <= val_size:
            val_indices += index_set
            val_scaffold_count += 1
        else:
            test_indices += index_set
            test_scaffold_count += 1

    if logger is not None:
        logger.debug('Total scaffolds = {:,} | train scaffolds = {:,} | val scaffolds = {:,} | test scaffolds = {:,}'.format(
            len(scaffold_to_indices_map),
            train_scaffold_count,
            val_scaffold_count,
            test_scaffold_count
        ))
    
    log_scaffold_stats(data, index_sets, logger)

    train = [data[i] for i in train_indices]
    val = [data[i] for i in val_indices]
    test = [data[i] for i in test_indices]

    return train, val, test


def log_scaffold_stats(data, index_sets, logger=None):
    # print some statistics about scaffolds
    label_avgs = []
    counts = []
    for index_set in index_sets:
        data_set = [data[i] for i in index_set]
        _, labels = zip(*data_set)
        labels = np.array(labels, dtype=np.float)
        label_avgs.append(np.nanmean(labels, axis=0))
        counts.append(np.count_nonzero(~np.isnan(labels), axis=0))
    stats = [(label_avgs[i][:20], counts[i][:20]) for i in range(min(10, len(label_avgs)))]
    if logger is not None:
        logger.debug('Label averages per scaffold, in decreasing order of scaffold frequency, capped at 10 scaffolds and 20 labels: {}'.format(stats))
    return stats


def scaffold_split_one(data: List[Tuple[str, List[float]]]) -> Tuple[List[Tuple[str, List[float]]],
                                                                     List[Tuple[str, List[float]]],
                                                                     List[Tuple[str, List[float]]]]:
    """
    Split a dataset by scaffold such that train has all molecules from the largest scaffold
    (i.e. the scaffold with the most molecules), val has all molecules from the second largest
    scaffold, and test has all molecules from the third largest scaffold.

    :param data: A list of data points (smiles string, target values).
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    # Get data
    smiles, _ = zip(*data)
    scaffold_to_indices_map = scaffold_to_smiles(smiles, use_indices=True)

    # Sort from largest to smallest scaffold sets
    scaffolds = sorted(list(scaffold_to_indices_map.keys()),
                       key=lambda scaffold: len(scaffold_to_indices_map[scaffold]),
                       reverse=True)

    train = [data[index] for index in scaffold_to_indices_map[scaffolds[0]]]
    val = [data[index] for index in scaffold_to_indices_map[scaffolds[1]]]
    test = [data[index] for index in scaffold_to_indices_map[scaffolds[2]]]

    return train, val, test
