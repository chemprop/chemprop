from copy import deepcopy
import os
import pickle
import random

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)


class Args(Tap):
    save_dir: str  # Path to directory to save indices
    num_folds: int = 10  # Number of cross validation folds
    test_folds_to_test: int = None  # Number of cross validation folds to test as test folds
    val_folds_per_test: int = None  # Number of cross validation folds


def create_crossval_indices(args: Args):
    random.seed(0)
    if args.test_folds_to_test is None:
        args.test_folds_to_test = args.num_folds
    if args.val_folds_per_test is None:
        args.val_folds_per_test = args.num_folds - 1
    folds = list(range(args.num_folds))
    random.shuffle(folds)
    os.makedirs(args.save_dir, exist_ok=True)
    for i in folds[:args.test_folds_to_test]:
        with open(os.path.join(args.save_dir, f'{i}_opt.pkl'), 'wb') as valf, open(os.path.join(args.save_dir, f'{i}_test.pkl'), 'wb') as testf:
            index_sets = []
            test_index_sets = []
            index_folds = deepcopy(folds)
            index_folds.remove(i)
            random.shuffle(index_folds)
            for val_index in index_folds[:args.val_folds_per_test]:
                train, val, test = [index for index in index_folds if index != val_index], [val_index], [i]  # test set = val set during cv for now
                index_sets.append([train, val, val])
                test_index_sets.append([train, val, test])
            pickle.dump(index_sets, valf)
            pickle.dump(test_index_sets, testf)
        print(i, index_sets, test_index_sets)
        for j in range(len(index_sets)):
            with open(os.path.join(args.save_dir, 'mayr', f'{i}_{j}_opt.pkl'), 'wb') as valf, open(os.path.join(args.save_dir, 'mayr', f'{i}_{j}_test.pkl'), 'wb') as testf:
                pickle.dump(index_sets[j], valf)
                pickle.dump(test_index_sets[j], testf)


if __name__ == '__main__':
    create_crossval_indices(Args().parse_args())
