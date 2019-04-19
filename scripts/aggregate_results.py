from argparse import ArgumentParser, Namespace
import os
import numpy as np


def aggregate_results(args: Namespace):
    print(f'Walking {args.ckpts_dir} for verbose.log files')
    results = []
    for root, _, files in os.walk(args.ckpt_dir):
        for fname in files:
            if fname == 'verbose.log':
                with open(os.path.join(root, fname), 'r') as rf:
                    for line in rf:
                        last_line = line
                    # e.g. Overall test rmse = 0.939207 +/- 0.000000
                    last_line = last_line.strip().split('=')[1]
                    last_line = last_line.split('+')[0]
                    results.append(float(last_line.strip()))
    results = np.array(results)
    print(f'Mean: {np.mean(results)}, Std: {np.std(results)}, Total num files: {len(results)}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpts_dir', type=str, required=True,
                        help='Path to directory to with model save dirs')
    args = parser.parse_args()

    aggregate_results(args)
