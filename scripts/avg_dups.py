"""Averages the target values for duplicate smiles strings. (Only used for regression datasets.)"""

from collections import defaultdict
import os
import sys
from typing import List

import numpy as np
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_data, get_header


class Args(Tap):
    data_path: str  # Path to data CSV file
    smiles_columns: List[str] = None  # Name of the columns containing SMILES strings. By default, uses the first column.
    target_columns: List[str] = None  # Name of the columns containing target values. By default, uses all columns except the SMILES column.
    save_path: str  # Path where average data CSV file will be saved


def average_duplicates(args: Args):
    """Averages duplicate data points in a dataset."""
    print('Loading data')
    header = get_header(args.data_path)
    data = get_data(path=args.data_path, smiles_columns=args.smiles_columns, target_columns=args.target_columns)
    print(f'Data size = {len(data):,}')

    # Map SMILES string to lists of targets
    smiles_in_order = []
    smiles_to_targets = defaultdict(list)
    for smiles, targets in zip(data.smiles(flatten=True), data.targets()):
        smiles_to_targets[smiles].append(targets)
        if len(smiles_to_targets[smiles]) == 1:
            smiles_in_order.append(smiles)

    # Find duplicates
    duplicate_count = 0
    stds = []
    new_data = []
    for smiles in smiles_in_order:
        all_targets = smiles_to_targets[smiles]
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

    print(f'Number of duplicates = {duplicate_count:,}')
    print(f'Duplicate standard deviation per task = {", ".join(f":{std:.4e}" for std in np.mean(stds, axis=0))}')
    print(f'New data size = {len(new_data):,}')

    # Save new data
    with open(args.save_path, 'w') as f:
        f.write(','.join(header) + '\n')

        for smiles, avg_targets in new_data:
            f.write(smiles + ',' + ','.join(str(value) if value is not None else '' for value in avg_targets) + '\n')


if __name__ == '__main__':
    average_duplicates(Args().parse_args())
