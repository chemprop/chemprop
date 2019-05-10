from argparse import ArgumentParser
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data


def class_balance(data_path: str, split_type: str):
    # Update args
    args.val_fold_index, args.test_fold_index = 1, 2
    args.split_type = 'predetermined'

    # Load data
    data = get_data(path=args.data_path)
    args.task_names = get_task_names(path=args.data_path)

    for i in range(10):
        # Update args
        data_name = os.path.splitext(os.path.basename(data_path))[0]
        args.folds_file = f'/data/rsg/chemistry/yangk/lsc_experiments_dump_splits/data/{data_name}/{split_type}/fold_{i}/0/split_indices.pckl'

        if not os.path.exists(args.folds_file):
            print(f'Fold indices for fold {i} not found')
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV file')
    parser.add_argument('--split_type', type=str, default='scaffold',
                        help='Method of splitting data')
    args = parser.parse_args()

    class_balance(
        data_path=args.data_path,
        split_type=args.split_type
    )
