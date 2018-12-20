from argparse import ArgumentParser, Namespace
from collections import Counter
from multiprocessing import Pool
from functools import partial
import os
import sys
sys.path.append('../')
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from chemprop.data.utils import get_data
from chemprop.data.vocab import atom_vocab


def count_vocab(pair: Tuple[Callable, str]) -> Counter:
    vocab_func, smiles = pair
    return Counter(vocab_func(smiles))


def plot_counts(counter: Counter, args: Namespace):
    data_name = os.path.basename(args.data_path).replace('.csv', '')
    total_count = sum(counter.values())

    for num_to_plot in args.nums_to_plot:
        plt.clf()

        _, values = zip(*counter.most_common(num_to_plot))
        indexes = np.arange(len(values))

        plt.bar(indexes, values, width=1)
        plt.title('{} {} frequency'.format(data_name, args.vocab_func))
        plt.xlabel('{} 100 most common {}s'.format(num_to_plot, args.vocab_func))
        plt.ylabel('frequency')

        if args.plot_dir is not None:
            plt.savefig(os.path.join(args.plot_dir, '{}_frequency.png'.format(num_to_plot)))
        else:
            plt.show()

        plt.clf()

        # Plot cumulative frequency distribution
        cumulative_counts = np.cumsum(values)
        cumulative_freqs = cumulative_counts / total_count

        plt.bar(indexes, cumulative_freqs, width=1)
        plt.title('{} {} cumulative frequency'.format(data_name, args.vocab_func))
        plt.xlabel('{} most common {}s'.format(num_to_plot, args.vocab_func))
        plt.ylabel('cumulative')

        if args.plot_dir is not None:
            plt.savefig(os.path.join(args.plot_dir, '{}_cumulative_frequency.png'.format(num_to_plot)))
        else:
            plt.show()


def generate_vocab(args: Namespace):
    # Get smiles
    data = get_data(args.data_path)
    smiles = data.smiles()

    vocab_func = partial(
        atom_vocab,
        vocab_func=args.vocab_func,
        substructure_sizes=args.substructure_sizes
    )

    pairs = [(vocab_func, smile) for smile in smiles]

    if args.sequential:
        counter = sum([count_vocab(pair) for pair in tqdm(pairs, total=len(pairs))], Counter())
    else:
        with Pool() as pool:
            counter = sum(pool.map(count_vocab, pairs), Counter())

    print('Vocab size = {:,}'.format(len(counter)))

    # Save vocab
    if args.vocab_path is not None:
        with open(args.vocab_path, 'w') as f:
            for word in counter.keys():
                f.write(word + '\n')

    # Save vocab with counts
    if args.counts_path is not None:
        with open(args.counts_path, 'w') as f:
            for word, count in counter.most_common():
                f.write(word + ',' + str(count) + '\n')

    # Plot vocab frequency distributions
    plot_counts(counter, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file')
    parser.add_argument('--vocab_func', type=str, required=True,
                        choices=['atom', 'atom_features', 'feature_vector', 'substructure'],
                        help='Type of vocabulary to generate')
    parser.add_argument('--substructure_sizes', type=int, nargs='+', default=[3],
                        help='Substructure sizes when using vocab_func "substructure"')
    parser.add_argument('--vocab_path', type=str,
                        help='Path where vocab will be saved')
    parser.add_argument('--counts_path', type=str,
                        help='Path where vocab with counts will be saved as a CSV')
    parser.add_argument('--plot_dir', type=str,
                        help='Directory where vocab frequency plots will be saved')
    parser.add_argument('--nums_to_plot', type=int, nargs='+', default=[100, 200, 500, 1000],
                        help='X most common to plot')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Whether to run sequentially instead of in parallel')
    args = parser.parse_args()

    if args.plot_dir is not None:
        os.makedirs(args.plot_dir, exist_ok=True)

    generate_vocab(args)
