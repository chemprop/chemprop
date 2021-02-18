"""Cleans a dataset by removing molecules which cannot be parsed by RDKit."""

import csv
from rdkit import Chem

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)


class Args(Tap):
    data_path: str  # Data CSV to sanitize
    save_path: str  # Path to CSV where sanitized data will be saved


def sanitize(data_path: str, save_path: str):
    with open(data_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        lines = [line for line in reader if line[0] != '' and Chem.MolFromSmiles(line[0]) is not None]

    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for line in lines:
            writer.writerow(line)


if __name__ == '__main__':
    args = Args().parse_args()

    sanitize(args.data_path, args.save_path)
