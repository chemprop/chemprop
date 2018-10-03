"""Averages duplicate data points in a dataset."""

from argparse import ArgumentParser

import numpy as np

from utils import get_data_with_header


def average_duplicates(args):
    print('Loading data')
    header, data = get_data_with_header(args.data_path)
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
        stds.append(np.std(all_values, axis=0))

        avg_values = np.mean(all_values, axis=0)
        new_data.append((smiles, avg_values))

    print('Number of duplicates = {:,}'.format(duplicate_count))
    print('Duplicate standard deviation per task = {}'.format(', '.join('{:.4e}'.format(std) for std in np.mean(stds, axis=0))))
    print('New data size = {:,}'.format(len(new_data)))

    # Save new data
    with open(args.save_path, 'w') as f:
        f.write(','.join(header) + '\n')

        for smiles, avg_values in new_data:
            f.write(smiles + ','.join(str(value) for value in avg_values) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--save_path', type=str,
                        help='Path where average data CSV file will be saved')
    args = parser.parse_args()

    average_duplicates(args)
