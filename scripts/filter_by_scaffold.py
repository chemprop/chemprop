from argparse import ArgumentParser, Namespace
from collections import namedtuple
import sys
sys.path.append('../')
from typing import List

from tqdm import tqdm

from chemprop.data.scaffold import generate_scaffold


Datapoint = namedtuple('Datapoint', ['smiles', 'line'])


def get_header(path: str) -> str:
    with open(path) as f:
        header = f.readline()

    return header


def get_data(path: str) -> List[Datapoint]:
    with open(path) as f:
        f.readline()  # skip header
        data = []
        for line in f.readlines():
            smiles = line[:line.index(',')]
            data.append(Datapoint(smiles=smiles, line=line))

    return data


def filter_by_scaffold(args: Namespace):
    print('Loading data')
    header = get_header(args.data_path)
    data = get_data(args.data_path)
    scaffold_data = get_data(args.scaffold_data_path)

    print('Generating scaffolds')
    smiles_to_scaffold = {d.smiles: generate_scaffold(d.smiles) for d in tqdm(data, total=len(data))}
    scaffolds_to_keep = {generate_scaffold(d.smiles) for d in tqdm(scaffold_data, total=len(scaffold_data))}

    print('Filtering data')
    filtered_data = [d for d in data if smiles_to_scaffold[d.smiles] in scaffolds_to_keep]

    print(f'Filtered data from {len(data):,} to {len(filtered_data):,} molecules')

    print('Saving data')
    with open(args.save_path, 'w') as f:
        f.write(header)
        for d in filtered_data:
            f.write(d.line)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset which will be limited to those molecules sharing'
                             'a scaffold with a molecule in the scaffold_data_path dataset')
    parser.add_argument('--scaffold_data_path', type=str, required=True,
                        help='Path to the dataset whose scaffolds will be used to limit the'
                             'molecules in data_path')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path where the filtered version of data_path will be saved')
    args = parser.parse_args()

    filter_by_scaffold(args)
