from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt

from utils import get_data_with_header


def plot_distribution(data_path: str, save_dir: str, bins: int):
    """
    Plots the distribution of values of a dataset.

    :param data_path: Path to data CSV file.
    :param save_dir: Directory where plot PNGs will be saved.
    :param bins: Number of bins in histogram.
    """
    # Get values
    header, data = get_data_with_header(data_path)
    task_names = header[1:]
    _, values = zip(*data)

    # Arrange values by task
    data_size, num_tasks = len(values), len(task_names)
    values = [[values[i][j] for i in range(data_size)] for j in range(num_tasks)]

    # Plot distributions for each task
    data_name = os.path.basename(data_path).replace('.csv', '')

    for i in range(num_tasks):
        plt.clf()
        plt.hist(values[i], bins=bins)

        # Save plot
        plt.title('{} - {}'.format(data_name, task_names[i]))
        plt.xlabel(task_names[i])
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(save_dir, '{}_{}.png'.format(data_name, task_names[i])))


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
