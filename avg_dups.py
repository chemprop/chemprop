"""Averages duplicate data points in a dataset."""

from argparse import ArgumentParser

import numpy as np

from utils import get_data


def average_duplicates(args):
    print('Loading data')
    header, data = get_data(args.data_path, get_header=True)
    print('Data size = {:,}'.format(len(data)))

    # Map SMILES string to lists of values
    smiles_to_values = {}
    for smiles, values in data:
        smiles_to_values.setdefault(smiles, []).append(values)

    # Find duplicates
    duplicate_count = 0
    stds = []
    new_data = []
    for smiles, all_values in smiles_to_values.items():
        duplicate_count += len(all_values) - 1
        num_tasks = len(all_values[0])

        values_by_task = [[] for _ in range(num_tasks)]
        for task in range(num_tasks):
            for values in all_values:
                if values[task] is not None:
                    values_by_task[task].append(values[task])

        stds.append([np.std(task_values) if len(task_values) > 0 else 0.0 for task_values in values_by_task])
        means = [np.mean(task_values) if len(task_values) > 0 else None for task_values in values_by_task]
        new_data.append((smiles, means))

    print('Number of duplicates = {:,}'.format(duplicate_count))
    print('Duplicate standard deviation per task = {}'.format(', '.join('{:.4e}'.format(std) for std in np.mean(stds, axis=0))))
    print('New data size = {:,}'.format(len(new_data)))

    # Save new data
    with open(args.save_path, 'w') as f:
        f.write(','.join(header) + '\n')

        for smiles, avg_values in new_data:
            f.write(smiles + ',' + ','.join(str(value) if value is not None else '' for value in avg_values) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--save_path', type=str,
                        help='Path where average data CSV file will be saved')
    args = parser.parse_args()

    average_duplicates(args)
