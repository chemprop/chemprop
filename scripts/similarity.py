from argparse import ArgumentParser
import sys
sys.path.append('../')

from chemprop.data.utils import get_data
from chemprop.data import morgan_similarity, scaffold_similarity


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
    parser.add_argument('--similarity_measure', type=str, required=True, choices=['scaffold', 'morgan'],
                        help='Similarity measure to use to compare the two datasets')
    parser.add_argument('--radius', type=int, default=3,
                        help='Radius of Morgan fingerprint')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='Rate at which to sample pairs of molecules for Morgan similarity (to reduce time)')
    args = parser.parse_args()

    data_1 = get_data(args.data_path_1, use_compound_names=args.compound_names_1)
    data_2 = get_data(args.data_path_2, use_compound_names=args.compound_names_2)

    if args.similarity_measure == 'scaffold':
        scaffold_similarity(data_1.smiles(), data_2.smiles())
    elif args.similarity_measure == 'morgan':
        morgan_similarity(data_1.smiles(), data_2.smiles(), args.radius, args.sample_rate)
    else:
        raise ValueError('Similarity measure "{}" not supported.'.format(args.similarity_measure))
