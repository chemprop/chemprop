import pickle
from typing import List

import numpy as np


def get_features(path: str) -> List[np.ndarray]:
    with open(path, 'rb') as f:
        features = pickle.load(f)
    features = [np.array(feat.todense()) for feat in features]

    return features
