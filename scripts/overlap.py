from argparse import ArgumentParser
import sys
sys.path.append('../')

from chemprop.data.utils import get_data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path_1', type=str, required=True,
                        help='Path to first data CSV file')
    parser.add_argument('--data_path_2', type=str, required=True,
                        help='Path to second data CSV file')
    parser.add_argument('--compound_names_1', action='store_true', default=False,
                        help='Whether data_path_1 has compound names in addition to smiles')
    parser.add_argument('--compound_names_2', action='store_true', default=False,
                        help='Whether data_path_2 has compound names in addition to smiles')
    parser.add_argument('--save_intersection_path', type=str, default=None,
                        help='Path to save intersection at; labeled with data_path 1 header')
    args = parser.parse_args()

    data_1 = get_data(args.data_path_1, use_compound_names=args.compound_names_1)
    data_2 = get_data(args.data_path_2, use_compound_names=args.compound_names_2)

    smiles1 = set(data_1.smiles())
    smiles2 = set(data_2.smiles())
    size_1, size_2 = len(smiles1), len(smiles2)
    intersection = smiles1.intersection(smiles2)
    size_intersect = len(intersection)
    print('Size of dataset 1: {}'.format(size_1))
    print('Size of dataset 2: {}'.format(size_2))
    print('Size of intersection: {}'.format(size_intersect))
    print('Size of intersection as frac of dataset 1: {}'.format(size_intersect/size_1))
    print('Size of intersection as frac of dataset 2: {}'.format(size_intersect/size_2))

    if args.save_intersection_path is not None:
        with open(args.data_path_1, 'r') as rf, open(args.save_intersection_path, 'w') as wf:
            header = rf.readline()
            wf.write(header.strip() + '\n')
            for line in rf:
                if line.strip().split(',')[0] in intersection:
                    wf.write(line.strip() + '\n')
