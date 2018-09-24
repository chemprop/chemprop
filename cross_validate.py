import numpy as np

from parsing import get_parser, modify_args
from train import run_training


def cross_validate(args):
    # Run training on different random seeds for each fold
    test_scores = []
    for seed in range(args.num_folds):
        args.seed = seed
        test_scores.append(run_training(args))

    # Report results
    print('{}-fold cross validation'.format(args.num_folds))
    for seed, score in enumerate(test_scores):
        print('Seed {} ==> test {} = {:.3f}'.format(seed, args.metric, score))
    print('Overall test {} = {:.3f} ± {:.3f}'.format(args.metric, np.mean(test_scores), np.std(test_scores)))


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds when performing cross validation')
    args = parser.parse_args()
    modify_args(args)

    cross_validate(args)
