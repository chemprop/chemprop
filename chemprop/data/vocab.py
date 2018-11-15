from multiprocessing import Pool
from typing import Callable, List, Set

from rdkit import Chem

from chemprop.features import atom_features


def atom_vocab(smiles: str) -> Set[str]:
    return {str(atom.GetAtomicNum()) for atom in Chem.MolFromSmiles(smiles).GetAtoms()}


def atom_features_vocab(smiles: str) -> Set[str]:
    return {str(atom_features(atom)) for atom in Chem.MolFromSmiles(smiles).GetAtoms()}


def parallel_vocab(vocab_func: Callable, smiles: List[str]) -> Set[str]:
    return set.union(*Pool().map(vocab_func, smiles))
