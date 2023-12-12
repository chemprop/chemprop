import csv
from pathlib import Path

import numpy as np
import pytest

from chemprop.v2.data import (
    MoleculeDataset, MoleculeDatapoint, SeededSampler, ClassBalanceSampler
)
from chemprop.v2.featurizers import MoleculeMolGraphFeaturizer


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


@pytest.fixture(params=[0.0, 0.1, 0.5, 1.0])
def t(request):
    return request.param


@pytest.fixture
def targets(scores, t):
    return np.where(scores <= t, 0, 1).astype(bool)


@pytest.fixture
def featurizer():
    return MoleculeMolGraphFeaturizer()


@pytest.fixture
def dataset(smis, scores, featurizer):
    data = [MoleculeDatapoint(smi, score) for smi, score in zip(smis, scores)]

    return MoleculeDataset(data, featurizer)


@pytest.fixture(params=[0, 24, 100])
def seed(request):
    return request.param


@pytest.fixture
def class_sampler(smis, targets, featurizer):
    data = [MoleculeDatapoint(smi, target) for smi, target in zip(smis, targets)]
    dset = MoleculeDataset(data, featurizer)

    return ClassBalanceSampler(dset, shuffle=True)


def test_seeded_no_seed(dataset):
    with pytest.raises(ValueError):
        SeededSampler(dataset, None)


def test_seeded_shuffle(dataset, seed):
    sampler = SeededSampler(dataset, seed)

    assert list(sampler) != list(sampler)


def test_seeded_fixed_shuffle(dataset, seed):
    sampler1 = SeededSampler(dataset, seed)
    sampler2 = SeededSampler(dataset, seed)

    idxs1 = list(sampler1)
    idxs2 = list(sampler2)

    assert idxs1 == idxs2


def test_class_balance_length(class_sampler, targets: np.ndarray):
    n_actives = targets.any(1).sum(0)
    n_inactives = len(targets) - n_actives
    expected_length = 2 * min(n_actives, n_inactives)

    assert len(class_sampler) == expected_length


def test_class_balance_sample(class_sampler, targets: np.ndarray):
    idxs = list(class_sampler)

    # sampled indices should be 50/50 actives/inacitves
    assert sum(targets[idxs]) == len(idxs) // 2


def test_class_balance_shuffle(class_sampler):
    idxs1 = list(class_sampler)
    idxs2 = list(class_sampler)

    if len(class_sampler) == 0:
        pytest.skip("no indices to sample!")

    assert idxs1 != idxs2


def test_seed_class_balance_shuffle(smis, targets, featurizer, seed):
    data = [MoleculeDatapoint(smi, target) for smi, target in zip(smis, targets)]
    dset = MoleculeDataset(data, featurizer)

    sampler = ClassBalanceSampler(dset, seed, True)

    if len(sampler) == 0:
        pytest.skip("no indices to sample!")

    assert list(sampler) != list(sampler)


def test_seed_class_balance_reproducibility(smis, targets, featurizer, seed):
    data = [MoleculeDatapoint(smi, target) for smi, target in zip(smis, targets)]
    dset = MoleculeDataset(data, featurizer)

    sampler1 = ClassBalanceSampler(dset, seed, True)
    sampler2 = ClassBalanceSampler(dset, seed, True)

    assert list(sampler1) == list(sampler2)
