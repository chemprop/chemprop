from argparse import ArgumentParser, Namespace
from collections import Counter
from multiprocessing import Pool
import os
import sys
from typing import List
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from tqdm import tqdm

from chemprop.data.utils import get_data
from chemprop.features.featurization import atom_features, bond_features
from chemprop.models.jtnn import MolTree


def junction_tree_node_words(smiles: str) -> List[str]:
    """
    Given a molecule smiles string, returns a list of junction tree node smiles strings.

    :param smiles: A smiles string representing a molecule.
    :return: A list of smiles strings representing the junction tree nodes.
    """
    mol = MolTree(smiles)
    words = [node.smiles for node in mol.nodes]

    return words


def bond_words(smiles: str) -> List[str]:
    """
    Given a molecule smiles string, returns a list of bond strings "atomic_num_1-bond_type-atomic_num_2".

    :param smiles: A smiles string representing a molecule.
    :return: A list of strings representing the bonds.
    """
    mol = Chem.MolFromSmiles(smiles)

    words = []
    for bond in mol.GetBonds():
        a1, bt, a2 = bond.GetBeginAtom().GetAtomicNum(), bond.GetBondType(), bond.GetEndAtom().GetAtomicNum()
        words.append('{}-{}-{}'.format(a1, bt, a2))
        words.append('{}-{}-{}'.format(a2, bt, a1))

    return words


def bond_features_words(smiles: str) -> List[str]:
    """
    Given a molecule smiles string, returns a list of bond strings where each bond string
    is the features of the two atoms "(atom_1_features_tuple)-(bond_features_tuple)-(atom_2_features_tuple)".

    :param smiles: A smiles string representing a molecule.
    :return: A list of strings representing the bonds with atom features.
    """
    mol = Chem.MolFromSmiles(smiles)

    words = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        a1_f, b_f, a2_f = atom_features(a1), bond_features(bond), atom_features(a2)
        a1_f, b_f, a2_f = tuple(a1_f), tuple(b_f), tuple(a2_f)
        words.append('{}-{}-{}'.format(a1_f, b_f, a2_f))
        words.append('{}-{}-{}'.format(a2_f, b_f, a1_f))

    return words


def generate_vocab(args: Namespace):
    # Get smiles
    data = get_data(args.data_path)
    smiles = data.smiles()

    # Determine vocab func
    if args.vocab_type == 'junction_tree_node':
        words_func = junction_tree_node_words
    elif args.vocab_type == 'bond':
        words_func = bond_words
    elif args.vocab_type == 'bond_features':
        words_func = bond_features_words
    else:
        raise ValueError('Vocab type "{}" not supported.'.format(args.vocab_type))

    # Create vocabs
    if args.sequential:
        words = [word for smile in tqdm(smiles, total=len(smiles)) for word in words_func(smile)]
    else:
        words = [word for words in Pool().map(words_func, smiles) for word in words]

    # Get vocab counts
    vocab_counts = Counter(words)

    # Save vocab
    with open(args.vocab_path, 'w') as f:
        for word in vocab_counts.keys():
            f.write(word + '\n')

    # Save vocab with counts
    with open(args.counts_path, 'w') as f:
        for word, count in vocab_counts.most_common():
            f.write(word + ',' + str(count) + '\n')

    # Plot vocab frequency distribution
    if args.plot_path is not None:
        _, values = zip(*vocab_counts.most_common(100))
        indexes = np.arange(len(values))

        plt.bar(indexes, values, width=1)
        plt.title(os.path.basename(args.data_path).replace('.csv', '') + ' {} frequency'.format(args.vocab_type))
        plt.xlabel('100 most common {}s'.format(args.vocab_type))
        plt.ylabel('frequency')
        plt.savefig(args.plot_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file')
    parser.add_argument('--vocab_type', type=str, required=True,
                        choices=['junction_tree_node', 'bond', 'bond_features'],
                        help='Type of vocabulary to generate')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Path where vocab will be saved')
    parser.add_argument('--counts_path', type=str, required=True,
                        help='Path where vocab with counts will be saved as a CSV')
    parser.add_argument('--plot_path', type=str, default=None,
                        help='Path where vocab frequency plot will be saved')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Whether to run sequentially instead of in parallel')
    args = parser.parse_args()

    generate_vocab(args)
