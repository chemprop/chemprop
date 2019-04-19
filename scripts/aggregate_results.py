from argparse import ArgumentParser
import os
from typing import List

import numpy as np


def aggregate_results(ckpts_dirs: List[str]):
    names = [os.path.basename(ckpts_dir) for ckpts_dir in ckpts_dirs]
    
    means, stds = [], []
    for ckpts_dir in ckpts_dirs:
        print(f'Walking {ckpts_dir} for verbose.log files')
        results = []

        # Collect verbose.log files
        paths = []
        for root, _, files in os.walk(ckpts_dir):
            paths += [os.path.join(root, fname) for fname in files if fname == 'verbose.log']

        # Process verbose.log files
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
                    break

        if invalid:
            print('Invalid verbose.log file')
            means.append('N/A')
            stds.append('N/A')
            continue

        # Compute results
        mean, std = np.mean(results), np.std(results)
        print(f'Mean: {mean}, Std: {std}, Total num files: {len(results)}')
        means.append(mean)
        stds.append(std)

    print()
    print('Results')
    print('Mean\tStd')
    for name, mean, std in zip(names, means, stds):
        print(f'{name}\t{mean}\t{std}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpts_dirs', type=str, nargs='+', required=True,
                        help='Path to directories (one per dataset) with model save dirs')
    args = parser.parse_args()

    aggregate_results(
        ckpts_dirs=args.ckpts_dirs
    )
