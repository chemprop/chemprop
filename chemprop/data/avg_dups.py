"""Averages duplicate data points in a dataset."""

from collections import defaultdict

import numpy as np

from chemprop.data.utils import get_data, get_header


def average_duplicates(args):
    print('Loading data')
    header = get_header(args.data_path)
    data = get_data(args.data_path)
    print('Data size = {:,}'.format(len(data)))

    # Map SMILES string to lists of targets
    smiles_to_targets = defaultdict(list)
    for smiles, targets in zip(data.smiles(), data.targets()):
        smiles_to_targets[smiles].append(targets)

    # Find duplicates
    duplicate_count = 0
    stds = []
    new_data = []
    for smiles, all_targets in smiles_to_targets.items():
        duplicate_count += len(all_targets) - 1
        num_tasks = len(all_targets[0])

        targets_by_task = [[] for _ in range(num_tasks)]
        for task in range(num_tasks):
            for targets in all_targets:
                if targets[task] is not None:
                    targets_by_task[task].append(targets[task])

        stds.append([np.std(task_targets) if len(task_targets) > 0 else 0.0 for task_targets in targets_by_task])
        means = [np.mean(task_targets) if len(task_targets) > 0 else None for task_targets in targets_by_task]
        new_data.append((smiles, means))

    print('Number of duplicates = {:,}'.format(duplicate_count))
    print('Duplicate standard deviation per task = {}'.format(', '.join('{:.4e}'.format(std) for std in np.mean(stds, axis=0))))
    print('New data size = {:,}'.format(len(new_data)))

    # Save new data
    with open(args.save_path, 'w') as f:
        f.write(','.join(header) + '\n')

        for smiles, avg_targets in new_data:
            f.write(smiles + ',' + ','.join(str(value) if value is not None else '' for value in avg_targets) + '\n')
