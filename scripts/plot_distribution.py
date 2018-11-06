from argparse import ArgumentParser
import os

from chemprop.data_processing import plot_distribution


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--save_dir', type=str,
                        help='Directory where plot PNGs will be saved')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of bins in histogram.')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    plot_distribution(args.data_path, args.save_dir, args.bins)
