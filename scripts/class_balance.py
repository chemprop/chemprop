import os
import sys
from typing import List
from typing_extensions import Literal


import numpy as np
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from chemprop.data import get_class_sizes, get_data, get_task_names, split_data


class Args(Tap):
    data_path: str  # Path to data CSV file
    smiles_columns: List[str] = None  # Name of the columns containing SMILES strings. By default, uses the first column.
    target_columns: List[str] = None  # Name of the columns containing target values. By default, uses all columns except the SMILES column.
    split_type: Literal['random', 'scaffold'] = 'scaffold'  # Split type, either "random" or "scaffold"


def class_balance(args: Args):
    # Update args
    args.val_fold_index, args.test_fold_index = 1, 2
    args.split_type = 'predetermined'

    # Load data
    data = get_data(path=args.data_path, smiles_columns=args.smiles_columns, target_columns=args.target_columns)
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns, target_columns=args.target_columns)

    # Average class sizes
    all_class_sizes = {
        'train': [],
        'val': [],
        'test': []
    }

    for i in range(10):
        print(f'Fold {i}')

        # Update args
        data_name = os.path.splitext(os.path.basename(args.data_path))[0]
        args.folds_file = f'/data/rsg/chemistry/yangk/lsc_experiments_dump_splits/data/{data_name}/{args.split_type}/fold_{i}/0/split_indices.pckl'

        if not os.path.exists(args.folds_file):
            print('Fold indices do not exist')
            continue

        # Split data
        train_data, val_data, test_data = split_data(
            data=data,
            split_type=args.split_type,
            args=args
        )

        # Determine class balance
        for data_split, split_name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            class_sizes = get_class_sizes(data_split)
            print(f'Class sizes for {split_name}')

            for i, task_class_sizes in enumerate(class_sizes):
                print(f'{args.task_names[i]} '
                      f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

            all_class_sizes[split_name].append(class_sizes)

        print()

    # Mean and std across folds
    for split_name in ['train', 'val', 'test']:
        print(f'Average class sizes for {split_name}')

        mean_class_sizes, std_class_sizes = np.mean(all_class_sizes[split_name], axis=0), np.std(all_class_sizes[split_name], axis=0)

        for i, (mean_task_class_sizes, std_task_class_sizes) in enumerate(zip(mean_class_sizes, std_class_sizes)):
            print(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {mean_size * 100:.2f}% +/- {std_size * 100:.2f}%" for cls, (mean_size, std_size) in enumerate(zip(mean_task_class_sizes, std_task_class_sizes)))}')


if __name__ == '__main__':
    class_balance(args=Args().parse_args())
