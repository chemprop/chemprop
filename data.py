import random
from typing import List

import numpy as np
from torch.utils.data.dataset import Dataset

from morgan_fingerprint import morgan_fingerprint


class MoleculeDatapoint:
    def __init__(self,
                 line: List[str],
                 features: np.ndarray = None,
                 features_generator: str = None,
                 use_compound_names: bool = False):
        if features is not None and features_generator is not None:
            raise ValueError('Currently cannot provide both loaded features and a features generator.')

        if use_compound_names:
            self.compound_name = line[0]  # str
            line = line[1:]
        else:
            self.compound_name = None

        self.smiles = line[0]  # str
        self.features = features  # np.ndarray
        self.targets = [float(x) if x != '' else None for x in line[1:]]  # List[Optional[float]]
        self.num_tasks = len(self.targets)  # int

        # Generate additional features
        if features_generator is not None:
            if features_generator == 'morgan':
                self.features = morgan_fingerprint(self.smiles)  # np.ndarray
            else:
                raise ValueError('features_generator type "{}" not supported.'.format(features_generator))


class MoleculeDataset(Dataset):
    def __init__(self, data: List[MoleculeDatapoint]):
        self.data = data

    def compound_names(self):
        return [d.compound_name for d in self.data]

    def smiles(self):
        return [d.smiles for d in self.data]

    def features(self):
        return [d.features for d in self.data]

    def targets(self):
        return [d.targets for d in self.data]

    def num_tasks(self):
        return len(self.data[0].targets)

    def shuffle(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
