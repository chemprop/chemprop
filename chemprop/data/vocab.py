from argparse import Namespace
from multiprocessing import Pool
from typing import Callable, List, Set, Tuple

from rdkit import Chem

from chemprop.features import atom_features


def atom_vocab(smiles: str) -> List[str]:
    return [str(atom.GetAtomicNum()) for atom in Chem.MolFromSmiles(smiles).GetAtoms()]


def atom_features_vocab(smiles: str) -> List[str]:
    return [str(atom_features(atom)) for atom in Chem.MolFromSmiles(smiles).GetAtoms()]


def vocab(pair: Tuple[Callable, str]) -> Set[str]:
    vocab_func, smiles = pair
    return set(vocab_func(smiles))


def parallel_vocab(vocab_func: Callable, smiles: List[str]) -> Set[str]:
    pairs = [(vocab_func, smile) for smile in smiles]
    return set.union(*Pool().map(vocab, pairs))


def get_vocab_func(args: Namespace) -> Callable:
    vocab_func = args.bert_vocab_func

    if vocab_func == 'atom':
        return atom_vocab

    if vocab_func == 'atom_features':
        return atom_features_vocab

    raise ValueError('Vocab function "{}" not supported.'.format(vocab_func))
