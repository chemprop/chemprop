import os
from typing import List
from typing_extensions import Literal


from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)


class Args(Tap):
    ckpts_dirs: List[str]  # Paths to directory containing hyperopt config.json files in directories labelled by fold number (0, 1, ...)
    split_type: Literal['random', 'scaffold']  # Split type, either "random" or "scaffold"
    num_folds: int = 10  # Number of folds


def main(ckpts_dirs: List[str], split_type: str, num_folds: int):
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
    args = Args().parse_args()

    main(
        ckpts_dirs=args.ckpts_dirs,
        split_type=args.split_type,
        num_folds=args.num_folds
    )
