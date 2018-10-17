from argparse import ArgumentParser
import logging

from scaffold import scaffold_split
from utils import get_data, get_header


# Initialize logger
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV file')
    parser.add_argument('--train_save', type=str, required=True,
                        help='Path where train CSV will be saved')
    parser.add_argument('--val_save', type=str, required=True,
                        help='Path where validation CSV will be saved')
    parser.add_argument('--test_save', type=str, required=True,
                        help='Path where test CSV will be saved')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    args = parser.parse_args()

    # Split data
    data = get_data(args.data_path)
    train, val, test = scaffold_split(data=data, sizes=args.split_sizes, logger=logger)

    # Save splits
    header = get_header(args.data_path)
    for fname, data in [(args.train_save, train), (args.val_save, val), (args.test_save, test)]:
        with open(fname, 'w') as f:
            f.write(','.join(header) + '\n')
            for smiles, labels in data:
                f.write(smiles + ',' + ','.join(str(l) if l is not None else '' for l in labels) + '\n')
