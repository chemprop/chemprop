from argparse import Namespace
from multiprocessing import Pool
from typing import Callable, List, Set, Tuple, Union
from functools import partial

from rdkit import Chem
import torch

from chemprop.features import atom_features, ATOM_FDIM


class Vocab:
    def __init__(self, args: Namespace, smiles: List[str]):
        self.vocab_func = get_vocab_func(args)
        if args.bert_vocab_func == 'atom':
            self.unk = '-1'
        if args.bert_vocab_func == 'atom_features':
            self.unk = str([0 for _ in range(ATOM_FDIM)])
        if args.bert_vocab_func == 'feature_vector':
            self.unk = None
            self.output_size = ATOM_FDIM
            return  # don't need a real vocab list here
        self.smiles = smiles
        self.vocab = parallel_vocab(self.vocab_func, self.smiles)
        self.vocab.add(self.unk)
        self.vocab_size = len(self.vocab)
        self.vocab_mapping = {word: i for i, word in enumerate(sorted(self.vocab))}
        self.output_size = self.vocab_size

    def w2i(self, word: str) -> int:
        if self.unk is None:
            return word  # in this case, we didn't map to a vocab at all; we're just predicting the original features
        return self.vocab_mapping[word] if word in self.vocab_mapping else self.vocab_mapping[self.unk]

    def smiles2indices(self, smiles: List[str]) -> Tuple[List[int], List[List[int]]]:
        features, nb_indices = self.vocab_func(smiles, nb_info=True)
        return [self.w2i(word) for word in features], nb_indices


def atom_vocab(smiles: str, vocab_func: str, nb_info: bool = False) -> Union[List[str],
                                                                             Tuple[List[str], List[List[int]]]]:
    if vocab_func == 'atom':
        featurizer = lambda x: x.GetAtomicNum()
    elif vocab_func == 'atom_features':
        featurizer = atom_features
    elif vocab_func == 'feature_vector':
        featurizer = atom_features
    else:
        raise ValueError('vocab_func "{}" not supported.'.format(vocab_func))

    all_atoms = Chem.MolFromSmiles(smiles).GetAtoms()
    if vocab_func == 'feature_vector':
        features = [featurizer(atom) for atom in all_atoms]
    else:
        features = [str(featurizer(atom)) for atom in all_atoms]

    if nb_info:
        nb_indices = []
        for atom in all_atoms:
            nb_indices.append([nb.GetIdx() for nb in atom.GetNeighbors()])  # atoms are sorted by idx

        return features, nb_indices

    return features


def vocab(pair: Tuple[Callable, str]) -> Set[str]:
    vocab_func, smiles = pair
    return set(vocab_func(smiles, nb_info=False))


def parallel_vocab(vocab_func: Callable, smiles: List[str]) -> Set[str]:
    pairs = [(vocab_func, smile) for smile in smiles]
    return set.union(*Pool().map(vocab, pairs))


def get_vocab_func(args: Namespace) -> Callable:
    vocab_func = args.bert_vocab_func

    if vocab_func in ['atom', 'atom_features', 'feature_vector']:
        return partial(atom_vocab, vocab_func=vocab_func)

    raise ValueError('Vocab function "{}" not supported.'.format(vocab_func))


def load_vocab(path: str) -> Vocab:
    """
    Loads the Vocab a model was trained with.

    :param path: Path where the model checkpoint is saved.
    :return: The Vocab object that the model was trained with.
    """
    return torch.load(path, map_location=lambda storage, loc: storage)['args'].vocab
