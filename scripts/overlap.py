"""Computes the overlap of molecules between two datasets."""

from argparse import ArgumentParser, Namespace
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_data


def overlap(args: Namespace):
    data_1 = get_data(path=args.data_path_1, use_compound_names=args.use_compound_names_1)
    data_2 = get_data(path=args.data_path_2, use_compound_names=args.use_compound_names_2)

    smiles1 = set(data_1.smiles())
    smiles2 = set(data_2.smiles())
    size_1, size_2 = len(smiles1), len(smiles2)
    intersection = smiles1.intersection(smiles2)
    size_intersect = len(intersection)
    print(f'Size of dataset 1: {size_1}')
    print(f'Size of dataset 2: {size_2}')
    print(f'Size of intersection: {size_intersect}')
    print(f'Size of intersection as frac of dataset 1: {size_intersect / size_1}')
    print(f'Size of intersection as frac of dataset 2: {size_intersect / size_2}')

    if args.save_intersection_path is not None:
        with open(args.data_path_1, 'r') as rf, open(args.save_intersection_path, 'w') as wf:
            reader, writer = csv.reader(rf), csv.writer(wf)
            header = next(reader)
            writer.writerow(header)
            for line in reader:
                if line[0] in intersection:
                    writer.writerow(line)

    if args.save_difference_path is not None:
        with open(args.data_path_1, 'r') as rf, open(args.save_difference_path, 'w') as wf:
            reader, writer = csv.reader(rf), csv.writer(wf)
            header = next(reader)
            writer.writerow(header)
            for line in reader():
                if line[0] not in intersection:
                    writer.writerow(line)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path_1', type=str, required=True,
                        help='Path to first data CSV file')
    parser.add_argument('--data_path_2', type=str, required=True,
                        help='Path to second data CSV file')
    parser.add_argument('--use_compound_names_1', action='store_true', default=False,
                        help='Whether data_path_1 has compound names in addition to smiles')
    parser.add_argument('--use_compound_names_2', action='store_true', default=False,
                        help='Whether data_path_2 has compound names in addition to smiles')
    parser.add_argument('--save_intersection_path', type=str, default=None,
                        help='Path to save intersection at; labeled with data_path 1 header')
    parser.add_argument('--save_difference_path', type=str, default=None,
                        help='Path to save molecules in dataset 1 that are not in dataset 2; labeled with data_path 1 header')
    args = parser.parse_args()

    overlap(args)
