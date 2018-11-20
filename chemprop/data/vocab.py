from argparse import Namespace
from multiprocessing import Pool
from typing import Callable, List, Set, Tuple
from functools import partial

from rdkit import Chem

from chemprop.features import atom_features, ATOM_FDIM


class Vocab:
    def __init__(self, args, smiles):
        self.vocab_func = get_vocab_func(args)
        if args.bert_vocab_func == 'atom':
            self.unk = '-1'
        if args.bert_vocab_func == 'atom_features':
            self.unk = str([0 for _ in range(ATOM_FDIM)])
        self.smiles = smiles
        self.vocab = parallel_vocab(self.vocab_func, self.smiles)
        self.vocab.add(self.unk)
        self.vocab_size = len(self.vocab)
        self.vocab_mapping = {word: i for i, word in enumerate(sorted(self.vocab))}

    def w2i(self, word):
        return self.vocab_mapping[word] if word in self.vocab_mapping else self.vocab_mapping[self.unk]

    def smiles2indices(self, smiles):
        features, nb_indices = self.vocab_func(smiles, nb_info=True)
        return [self.w2i(word) for word in features], nb_indices


def atom_vocab(smiles: str, vocab_func: str, nb_info: bool=False) -> List[str]:
    if vocab_func == 'atom':
        featurizer = lambda x: x.GetAtomicNum()
    elif vocab_func == 'atom_features':
        featurizer = atom_features
    all_atoms = Chem.MolFromSmiles(smiles).GetAtoms()
    features = [str(featurizer(atom)) for atom in all_atoms]
    if nb_info:
        nb_indices = []
        for atom in all_atoms:
            nb_indices.append([nb.GetIdx() for nb in atom.GetNeighbors()]) # atoms are sorted by idx
        return features, nb_indices
    else:
        return features


def vocab(pair: Tuple[Callable, str]) -> Set[str]:
    vocab_func, smiles = pair
    return set(vocab_func(smiles, nb_info=False))


def parallel_vocab(vocab_func: Callable, smiles: List[str]) -> Set[str]:
    pairs = [(vocab_func, smile) for smile in smiles]
    return set.union(*Pool().map(vocab, pairs))


def get_vocab_func(args: Namespace) -> Callable:
    vocab_func = args.bert_vocab_func

    if vocab_func == 'atom':
        return partial(atom_vocab, vocab_func='atom')

    if vocab_func == 'atom_features':
        return partial(atom_vocab, vocab_func='atom_features')

    raise ValueError('Vocab function "{}" not supported.'.format(vocab_func))
