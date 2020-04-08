import os
from typing import List
from typing_extensions import Literal


from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)
import numpy as np


ORDER = {
    name: index for index, name in enumerate([
        'qm7',
        'qm8',
        'qm9',
        'delaney',
        'freesolv',
        'lipo',
        'pdbbind_full',
        'pdbbind_core',
        'pdbbind_refined',
        'pcba',
        'muv',
        'hiv',
        'bace',
        'bbbp',
        'tox21',
        'toxcast',
        'sider',
        'clintox',
        'chembl',
        'rppb',
        'sol',
        'rlm',
        'hpxr',
        'hpxr (class)',
        'benzene',
        'cyclohexane',
        'dichloromethane',
        'dmso',
        'ethanol',
        'ethyl acetate',
        'h2o',
        'octanol',
        'tetrahydrofuran',
        'toluene',
        'logp'
    ])
}


class Args(Tap):
    ckpts_dirs: List[str]  # Path to directories (one per dataset) with model save dirs
    split_type: Literal['random', 'scaffold']  # Split type, either "random" or "scaffold"


def aggregate_results(ckpts_dirs: List[str], split_type: str):
    print('Name\tMean\tStd\tNum files')

    ckpts_dirs.sort(key=lambda ckpts_dir: ORDER[os.path.basename(ckpts_dir)])

    for ckpts_dir in ckpts_dirs:
        name = os.path.basename(ckpts_dir)

        # Collect verbose.log files
        paths = []
        for root, _, files in os.walk(ckpts_dir):
            if f'/{split_type}/' not in root:
                continue
            paths += [os.path.join(root, fname) for fname in files if fname == 'verbose.log']

        # Process verbose.log files
        results = []
        invalid = False
        for path in paths:
            with open(path) as rf:
                for line in rf:
                    last_line = line
                # e.g. Overall test rmse = 0.939207 +/- 0.000000
                try:
                    last_line = last_line.strip().split('=')[1]
                    last_line = last_line.split('+')[0]
                    results.append(float(last_line.strip()))
                except (IndexError, ValueError):
                    invalid = True

        if invalid:
            mean, std = 'N/A', 'N/A'
        else:
            mean, std = np.mean(results), np.std(results)

        # Compute results
        print(f'{name}\t{mean}\t{std}\t{len(results)}')


if __name__ == '__main__':
    args = Args().parse_args()

    aggregate_results(
        ckpts_dirs=args.ckpts_dirs,
        split_type=args.split_type
    )
