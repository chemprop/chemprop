import pickle
import os
from copy import deepcopy

folds = list(range(10))
os.makedirs('../crossval_index_files', exist_ok=True)
for i in folds:
    with open(os.path.join('../crossval_index_files', f'{i}.pkl'), 'wb') as wf:
        index_sets = []
        index_folds = deepcopy(folds)
        index_folds.remove(i)
        for val_index in index_folds:
            train, val, test = [index for index in index_folds if index != val_index], [val_index], [val_index] # test set = val set during cv for now
            index_sets.append([train, val, test])
        pickle.dump(index_sets, wf)