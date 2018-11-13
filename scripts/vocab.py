from argparse import ArgumentParser, Namespace
from collections import Counter
from multiprocessing import Pool
import sys
from typing import Set
sys.path.append('../')

import matplotlib.pyplot as plt
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

    # Plot vocab frequency distribution
    if args.plot_path is not None:
        _, values = zip(*vocab_counts.most_common())
        plt.hist(values, 100)
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Path where vocab will be saved')
    parser.add_argument('--plot_path', type=str, default=None,
                        help='Path where vocab frequency plot will be saved')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Whether to run sequentially instead of in parallel')
    args = parser.parse_args()

    generate_vocab(args)
