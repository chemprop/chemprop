from collections import defaultdict
from typing import Sequence

import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

from chemprop.v2.data.datapoints import DatapointBase


def split_data(
    data: Sequence[DatapointBase],
    split: str = "random",
    sizes: tuple[float, float, float] = (0.8, 0.1, 0.1),
    k: int = 5,
    fold: int = 0,
):
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Invalid train/val/test splits! got: {sizes}")

    n_train, n_val, n_test = [int(size * len(data)) for size in sizes]

    if split == "random":
        idxs = np.arange(len(data))
        np.random.shuffle(idxs)

        train = [data[i] for i in idxs[:n_train]]
        val = [data[i] for i in idxs[n_train:n_train + n_val]]
        test = [data[i] for i in idxs[n_train + n_val:]]
    elif split == "scaffold":
        scaffold2idxs = defaultdict(set)
        for i, d in enumerate(data):
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=d.mol)
            scaffold2idxs[scaffold].add(i)
            
        big_index_sets = []
        small_index_sets = []
        for idxs in scaffold2idxs.values():
            if len(idxs) > n_val / 2 or len(idxs) > n_test / 2:
                big_index_sets.append(idxs)
            else:
                small_index_sets.append(idxs)

        np.random.shuffle(big_index_sets)
        np.random.shuffle(small_index_sets)
        idx_sets = [*big_index_sets, *small_index_sets]

        train_idxs, val_idxs, test_idxs = [], [], []
        for idxs in idx_sets:
            if len(train) + len(idxs) <= n_train:
                train_idxs.extend(idxs)
                train_scaffold_count += 1
            elif len(val) + len(idxs) <= n_val:
                val_idxs.extend(idxs)
                val_scaffold_count += 1
            else:
                test_idxs.extend(idxs)
                test_scaffold_count += 1
        
        train = [data[i] for i in train_idxs]
        val = [data[i] for i in val_idxs]
        test = [data[i] for i in test_idxs]
    else:
        raise ValueError(f"Uknown split type! got: {split}")
    
    return train, val, test

