from argparse import Namespace
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
import random
from typing import Callable, List, FrozenSet, Set, Tuple, Union

from rdkit import Chem
import torch

from chemprop.features import atom_features, bond_features, get_atom_fdim, FunctionalGroupFeaturizer


class Vocab:
    def __init__(self, args: Namespace, smiles: List[str]):
        self.substructure_sizes = args.bert_substructure_sizes
        self.vocab_func = partial(
            atom_vocab,
            vocab_func=args.bert_vocab_func,
            substructure_sizes=self.substructure_sizes,
            to_set=True,
            args=args
        )

        if args.bert_vocab_func == 'feature_vector':
            self.unk = None
            self.output_size = get_atom_fdim(args, is_output=True)
            return  # don't need a real vocab list here

        self.unk = 'unk'
        self.smiles = smiles
        self.vocab = get_vocab(self.vocab_func, self.smiles, sequential=args.sequential)
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


def get_substructures_from_atom(atom: Chem.Atom,
                                max_size: int,
                                substructure: Set[int] = None) -> Set[FrozenSet[int]]:
    """
    Recursively gets all substructures up to a maximum size starting from an atom in a substructure.

    :param atom: The atom to start at.
    :param max_size: The maximum size of the substructure to fine.
    :param substructure: The current substructure that atom is in.
    :return: A set of substructures starting at atom where each substructure is a frozenset of indices.
    """
    assert max_size >= 1

    if substructure is None:
        substructure = {atom.GetIdx()}

    substructures = {frozenset(substructure)}

    if len(substructure) == max_size:
        return substructures

    # Get neighbors which are not already in the substructure
    new_neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetIdx() not in substructure]

    for neighbor in new_neighbors:
        # Define new substructure with neighbor
        new_substructure = deepcopy(substructure)
        new_substructure.add(neighbor.GetIdx())

        # Skip if new substructure has already been considered
        if frozenset(new_substructure) in substructures:
            continue

        # Recursively get substructures including this substructure plus neighbor
        new_substructures = get_substructures_from_atom(neighbor, max_size, new_substructure)

        # Add those substructures to current set of substructures
        substructures |= new_substructures

    return substructures


def get_substructures(atoms: List[Chem.Atom],
                      sizes: List[int],
                      max_count: int = None) -> Set[FrozenSet[int]]:
    """
    Gets up to max_count substructures (frozenset of atom indices) from a molecule.

    Note: Uses randomness to guarantee that the first max_count substructures
    found are a random sample of the substructures in the molecule.

    :param atoms: A list of atoms in the molecule.
    :param sizes: The sizes of substructures to find.
    :param max_count: The maximum number of substructures to find.
    :return: A set of substructures where each substructure is a frozenset of indices.
    """
    max_count = max_count or float('inf')

    random.shuffle(atoms)

    substructures = set()
    for atom in atoms:
        # Get all substructures up to max size starting from atom
        new_substructures = get_substructures_from_atom(atom, max(sizes))

        # Filter substructures to those which are one of the desired sizes
        new_substructures = [substructure for substructure in new_substructures if len(substructure) in sizes]

        for new_substructure in new_substructures:
            if len(substructures) >= max_count:
                break

            substructures.add(new_substructure)

    return substructures


def substructure_to_feature(mol: Chem.Mol,
                            substructure: FrozenSet[int],
                            fg_features: List[List[int]] = None) -> str:
    """
    Converts a substructure (set of atom indices) to a feature string
    by sorting and concatenating atom and bond feature vectors.

    :param mol: A molecule.
    :param substructure: A set of atom indices representing a substructure.
    :param fg_features: A list of k-hot vector indicating the functional groups the atom belongs to.
    :return: A string representing the featurization of the substructure.
    """
    if fg_features is None:
        fg_features = [None] * mol.GetNumAtoms()

    substructure = list(substructure)
    atoms = [Chem.Mol.GetAtomWithIdx(mol, idx) for idx in substructure]
    bonds = []
    for i in range(len(substructure)):
        for j in range(i + 1, len(substructure)):
            a1, a2 = substructure[i], substructure[j]
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is not None:
                bonds.append(bond)

    features = [str(atom_features(atom, fg_features[atom.GetIdx()])) for atom in atoms] + \
               [str(bond_features(bond)) for bond in bonds]
    features.sort()  # ensure identical feature string for different atom/bond ordering
    features = str(features)

    return features


def atom_vocab(smiles: str,
               vocab_func: str,
               args: Namespace,
               substructure_sizes: List[int] = None,
               to_set: bool = False,
               nb_info: bool = False) -> Union[List[str],
                                               Set[str],
                                               Tuple[List[str], List[List[int]]],
                                               Tuple[Set[str], List[List[int]]]]:
    if vocab_func not in ['atom', 'atom_features', 'feature_vector', 'substructure']:
        raise ValueError('vocab_func "{}" not supported.'.format(vocab_func))

    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()

    if 'functional_group' in args.additional_atom_features or 'functional_group' in args.additional_output_features:
        fg_featurizer = FunctionalGroupFeaturizer(args)
        fg_features = fg_featurizer.featurize(mol)
    else:
        fg_features = [None] * len(atoms)

    if vocab_func == 'feature_vector':
        features = [atom_features(atom, fg) for atom, fg in zip(atoms, fg_features)]
    elif vocab_func == 'atom_features':
        features = [str(atom_features(atom, fg)) for atom, fg in zip(atoms, fg_features)]
    elif vocab_func == 'atom':
        features = [str(atom.GetAtomicNum()) for atom in atoms]
    elif vocab_func == 'substructure':
        substructures = get_substructures(list(atoms), substructure_sizes)
        features = [substructure_to_feature(mol, substructure, fg_features) for substructure in substructures]
        import pdb; pdb.set_trace()
    else:
        raise ValueError('vocab_func "{}" not supported.'.format(vocab_func))

    if to_set and not vocab_func == 'feature_vector':
        features = set(features)

    if nb_info:
        nb_indices = []
        for atom in atoms:
            nb_indices.append([nb.GetIdx() for nb in atom.GetNeighbors()])  # atoms are sorted by idx

        return features, nb_indices

    return features


def vocab(pair: Tuple[Callable, str]) -> Set[str]:
    vocab_func, smiles = pair
    return set(vocab_func(smiles, nb_info=False))


def get_vocab(vocab_func: Callable, smiles: List[str], sequential: bool = False) -> Set[str]:
    pairs = [(vocab_func, smile) for smile in smiles]

    if sequential:
        return set.union(*map(vocab, pairs))

    return set.union(*Pool().map(vocab, pairs))


def load_vocab(path: str) -> Vocab:
    """
    Loads the Vocab a model was trained with.

    :param path: Path where the model checkpoint is saved.
    :return: The Vocab object that the model was trained with.
    """
    return torch.load(path, map_location=lambda storage, loc: storage)['args'].vocab
