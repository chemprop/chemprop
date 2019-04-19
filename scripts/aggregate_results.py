from argparse import ArgumentParser
import os
from typing import List

import numpy as np


def aggregate_results(ckpts_dirs: List[str]):
    print('Name\tMean\tStd\tNum files')

    for ckpts_dir in ckpts_dirs:
        name = os.path.basename(ckpts_dir)

        # Collect verbose.log files
        paths = []
        for root, _, files in os.walk(ckpts_dir):
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
                    break

        if invalid:
            mean, std = 'N/A', 'N/A'
        else:
            mean, std = np.mean(results), np.std(results)

        # Compute results
        print(f'{name}\t{mean}\t{std}\t{len(results)}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpts_dirs', type=str, nargs='+', required=True,
                        help='Path to directories (one per dataset) with model save dirs')
    args = parser.parse_args()

    aggregate_results(
        ckpts_dirs=args.ckpts_dirs
    )
