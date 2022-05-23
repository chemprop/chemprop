import csv
from pathlib import Path

import numpy as np
import pytest

from chemprop.data.v2.molecule import MolGraphDataset, MoleculeDatapoint
from chemprop.data.v2.sampler import SeededSampler, ClassBalanceSampler
from chemprop.featurizers.molgraph import MolGraphFeaturizer


TEST_DIR = Path(__file__).parents[2]
TEST_DATA_DIR = TEST_DIR / "data"


@pytest.fixture
def scores_csv():
    return TEST_DATA_DIR / "regression.csv"


@pytest.fixture
def smis(scores_csv):
    with open(scores_csv) as fid:
        reader = csv.reader(fid)
        next(reader)
        smis, _ = zip(*list(reader))

    return smis


@pytest.fixture
def scores(smis):
    return np.random.rand(len(smis), 1)


@pytest.fixture(params=[0., 0.1, 0.5, 1.])
def t(request):
    return request.param


@pytest.fixture
def targets(scores, t):
    return np.where(scores <= t, 0, 1)


@pytest.fixture
def featurizer():
    return MolGraphFeaturizer()


@pytest.fixture
def dataset(smis, scores, featurizer):
    data = [MoleculeDatapoint(smi, score) for smi, score in zip(smis, scores)]

    return MolGraphDataset(data, featurizer)


@pytest.fixture(params=[0, 24, 100])
def seed(request):
    return request.param


def test_seeded_no_seed(dataset):
    with pytest.raises(ValueError):
        SeededSampler(dataset, None)
    

def test_seeded_fixed_shuffle(dataset, seed):
    sampler = SeededSampler(dataset, seed)
    it1 = iter(sampler)
    it2 = iter(sampler)

    for idxs1, idxs2 in zip(it1, it2):
        assert idxs1 == idxs2


def test_class_balance_length(smis, targets: np.ndarray, featurizer):
    data = [MoleculeDatapoint(smi, target) for smi, target in zip(smis, targets)]

    n_actives = targets.astype(bool).any(1).sum(0)
    n_inactives = len(targets) - n_actives
    expected_length = 2 * min(n_actives, n_inactives)

    dset = MolGraphDataset(data, featurizer)
    sampler = ClassBalanceSampler(dset)
    
    assert len(sampler) == expected_length