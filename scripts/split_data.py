"""Splits data into train, validation, and test sets."""

import csv
import os
import sys
from typing import Tuple, List

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)
from tqdm import tqdm
from typing_extensions import Literal

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_smiles, MoleculeDatapoint, MoleculeDataset, split_data
from chemprop.utils import makedirs


class Args(Tap):
    data_path: str  # Path to data CSV file
    save_dir: str  # Directory where train, validation, and test sets will be saved
    smiles_columns: List[str] = None  # Name of the column containing SMILES strings. By default, uses the first column.
    split_type: Literal['random', 'scaffold_balanced'] = 'random'  # Split type
    split_sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # Split sizes
    seed: int = 0  # Random seed


def run_split_data(args: Args):
    # Load raw data
    with open(args.data_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        lines = list(reader)

    # Load SMILES
    smiles = get_smiles(path=args.data_path, smiles_columns=args.smiles_columns)

    # Make sure lines and smiles line up
    assert len(lines) == len(smiles)
    assert all(s in line for smile, line in zip(smiles, lines) for s in smile)

    # Create data
    data = []
    for smile, line in tqdm(zip(smiles, lines), total=len(smiles)):
        datapoint = MoleculeDatapoint(smiles=smile)
        datapoint.line = line
        data.append(datapoint)
    data = MoleculeDataset(data)

    train, val, test = split_data(
        data=data,
        split_type=args.split_type,
        sizes=args.split_sizes,
        seed=args.seed
    )

    makedirs(args.save_dir)

    for name, dataset in [('train', train), ('val', val), ('test', test)]:
        with open(os.path.join(args.save_dir, f'{name}.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for datapoint in dataset:
                writer.writerow(datapoint.line)


if __name__ == '__main__':
    run_split_data(Args().parse_args())
