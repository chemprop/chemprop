import os

import matplotlib.pyplot as plt

from chemprop.data.utils import get_data, get_task_names


def plot_distribution(data_path: str, save_dir: str, bins: int):
    """
    Plots the distribution of values of a dataset.

    :param data_path: Path to data CSV file.
    :param save_dir: Directory where plot PNGs will be saved.
    :param bins: Number of bins in histogram.
    """
    # Get values
    task_names = get_task_names(data_path)
    data = get_data(data_path)
    targets = data.targets()

    # Arrange values by task
    data_size, num_tasks = len(targets), len(task_names)
    values = [[targets[i][j] for i in range(data_size)] for j in range(num_tasks)]

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
