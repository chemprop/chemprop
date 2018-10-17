from collections import defaultdict
import logging
from typing import Dict, List, Set, Tuple, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


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
    scaffolds = scaffold_to_smiles(smiles, use_indices=True)  # mapping from scaffold to set of indices into smiles/data

    # Sort from largest to smallest scaffold sets
    index_sets = [sorted(list(index_set)) for index_set in scaffolds.values()]
    index_sets = sorted(index_sets, key=lambda index_set: len(index_set), reverse=True)

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
            len(scaffolds),
            train_scaffold_count,
            val_scaffold_count,
            test_scaffold_count
        ))

    train = [data[i] for i in train_indices]
    val = [data[i] for i in val_indices]
    test = [data[i] for i in test_indices]

    return train, val, test
