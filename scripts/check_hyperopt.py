from argparse import ArgumentParser
import os


def main(ckpts_dirs: str, split_type: str, num_folds: int):
    for ckpts_dir in ckpts_dirs:
        # Find all config.json files
        fnames = []
        for root, _, files in os.walk(ckpts_dir):
            if split_type not in root:
                continue
            fnames += [os.path.join(root, fname) for fname in files if fname == 'config.json']

        # Print out complete and incomplete
        complete = {int(os.path.basename(os.path.dirname(fname))) for fname in fnames}
        incomplete = set(range(num_folds)) - complete

        print(os.path.basename(ckpts_dir))
        print(f'complete = {" ".join(str(fold) for fold in sorted(complete))}')
        print(f'incomplete = {" ".join(str(fold) for fold in sorted(incomplete))}')
        print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpts_dirs', type=str, nargs='+', required=True,
                        help='Paths to directory containing hyperopt config.json files'
                             'in directories labelled by fold number (0, 1, ...)')
    parser.add_argument('--split_type', type=str, required=True,
                        help='"random" or "scaffold"')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='Number of folds')
    args = parser.parse_args()

    main(
        ckpts_dirs=args.ckpts_dirs,
        split_type=args.split_type,
        num_folds=args.num_folds
    )
