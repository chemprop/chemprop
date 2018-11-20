from argparse import Namespace
from multiprocessing import Pool
from typing import Callable, List, Set, Tuple

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
        return [self.w2i(word) for word in self.vocab_func(smiles)]


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
