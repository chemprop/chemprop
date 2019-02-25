"""Cleans a dataset by removing molecules which cannot be parsed by RDKit."""

from argparse import ArgumentParser
import csv
from rdkit import Chem


def sanitize(data_path: str, save_path: str):
    with open(data_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        lines = [line for line in reader if line[0] != '' and Chem.MolFromSmiles(line[0]) is not None]

    with open(save_path) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for line in lines:
            writer.writerow(line)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Data CSV to sanitize')
    parser.add_argument('--save_path', type=str, required=True, help='Path to CSV where sanitized data will be saved')
    args = parser.parse_args()

    sanitize(args.data_path, args.save_path)
