from argparse import ArgumentParser
import sys
sys.path.append('../')

from chemprop.data_processing import average_duplicates

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--save_path', type=str,
                        help='Path where average data CSV file will be saved')
    args = parser.parse_args()

    average_duplicates(args)
