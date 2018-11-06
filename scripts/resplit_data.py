from argparse import ArgumentParser
import os

from chemprop.data_processing import resplit


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to CSV file containing training data')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to CSV file containing val data')
    parser.add_argument('--train_save', type=str, required=True,
                        help='Path to CSV file for new train data')
    parser.add_argument('--val_save', type=str, required=True,
                        help='Path to CSV file for new val data')
    parser.add_argument('--val_frac', type=float, default=0.2,
                        help='frac of data to use for validation')
    args = parser.parse_args()

    # Create directory for save_path
    for path in [args.train_save, args.val_save]:
        save_dir = os.path.dirname(path)
        if save_dir != '':
            os.makedirs(save_dir, exist_ok=True)

    resplit(args)
