from argparse import ArgumentParser
import os


def main(ckpt_dir: str, num_folds: int):
    # Find all config.json files
    fnames = []
    for root, _, files in os.walk(ckpt_dir):
        fnames += [os.path.join(root, fname) for fname in files if fname == 'config.json']

    # Print out complete and incomplete
    complete = {int(os.path.basename(os.path.dirname(fname))) for fname in fnames}
    incomplete = set(range(num_folds)) - complete

    print(f'complete = {" ".join(sorted(complete))}')
    print(f'incomplete = {" ".join(sorted(incomplete))}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Path to directory containing hyperopt config.json files'
                             'in directories labelled by fold number (0, 1, ...)')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='Number of folds')
    args = parser.parse_args()

    main(
        ckpt_dir=args.ckpt_dir,
        num_folds=args.num_folds
    )
