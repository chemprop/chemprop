from argparse import ArgumentParser, Namespace
from collections import Counter
from multiprocessing import Pool
import os
import sys
from typing import Set
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from chemprop.data.utils import get_data
from chemprop.models.jtnn import MolTree


def vocab_for_mol(smiles: str) -> Set[str]:
    mol = MolTree(smiles)
    vocab = {node.smiles for node in mol.nodes}

    return vocab


def generate_vocab(args: Namespace):
    # Get smiles
    data = get_data(args.data_path)
    smiles = data.smiles()

    # Create vocabs
    if args.sequential:
        vocabs = [vocab_for_mol(s) for s in tqdm(smiles, total=len(smiles))]
    else:
        vocabs = Pool().map(vocab_for_mol, smiles)

    # Save vocab
    all_vocab = set()
    vocab_counts = Counter()
    with open(args.vocab_path, 'w') as f:
        for vocab in vocabs:
            vocab_counts.update(vocab)
            new_vocab = vocab - all_vocab
            for v in new_vocab:
                f.write(v + '\n')
            all_vocab |= new_vocab

    # Save vocab with counts
    with open(args.counts_path, 'w') as f:
        for label, value in vocab_counts.most_common():
            f.write(label + ',' + str(value) + '\n')

    # Plot vocab frequency distribution
    if args.plot_path is not None:
        _, values = zip(*vocab_counts.most_common(100))
        indexes = np.arange(len(values))

        plt.bar(indexes, values, width=1)
        plt.title(os.path.basename(args.data_path).replace('.csv', '') + ' junction tree node frequency')
        plt.xlabel('100 most common junction tree nodes')
        plt.ylabel('frequency')
        plt.savefig(args.plot_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file')
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
