import numpy as np
import pytest

from chemprop.data import ClassBalanceSampler, MoleculeDatapoint, MoleculeDataset, SeededSampler
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer


@pytest.fixture(params=[0.0, 0.1, 0.5, 1.0])
def threshold(request):
    return request.param


@pytest.fixture
def bin_targets(targets, threshold):
    return targets <= threshold


@pytest.fixture
def featurizer():
    return SimpleMoleculeMolGraphFeaturizer()


@pytest.fixture
def dataset(mols, targets, featurizer):
    data = [MoleculeDatapoint(mol, y) for mol, y in zip(mols, targets)]

    return MoleculeDataset(data, featurizer)


@pytest.fixture(params=[0, 24, 100])
def seed(request):
    return request.param


@pytest.fixture
def class_sampler(mols, bin_targets, featurizer):
    data = [MoleculeDatapoint(mol, y) for mol, y in zip(mols, bin_targets)]
    dset = MoleculeDataset(data, featurizer)

    return ClassBalanceSampler(dset.Y, shuffle=True)


def test_seeded_no_seed(dataset):
    with pytest.raises(ValueError):
        SeededSampler(len(dataset), None)


def test_seeded_shuffle(dataset, seed):
    sampler = SeededSampler(len(dataset), seed)

    assert list(sampler) != list(sampler)


def test_seeded_fixed_shuffle(dataset, seed):
    sampler1 = SeededSampler(len(dataset), seed)
    sampler2 = SeededSampler(len(dataset), seed)

    idxs1 = list(sampler1)
    idxs2 = list(sampler2)

    assert idxs1 == idxs2


def test_class_balance_length(class_sampler, bin_targets: np.ndarray):
    n_actives = bin_targets.any(1).sum(0)
    n_inactives = len(bin_targets) - n_actives
    expected_length = 2 * min(n_actives, n_inactives)

    assert len(class_sampler) == expected_length


def test_class_balance_sample(class_sampler, bin_targets: np.ndarray):
    idxs = list(class_sampler)

    # sampled indices should be 50/50 actives/inacitves
    assert sum(bin_targets[idxs]) == len(idxs) // 2


def test_class_balance_shuffle(class_sampler):
    idxs1 = list(class_sampler)
    idxs2 = list(class_sampler)

    if len(class_sampler) == 0:
        pytest.skip("no indices to sample!")

    assert idxs1 != idxs2


def test_seed_class_balance_shuffle(smis, bin_targets, featurizer, seed):
    data = [MoleculeDatapoint.from_smi(smi, target) for smi, target in zip(smis, bin_targets)]
    dset = MoleculeDataset(data, featurizer)

    sampler = ClassBalanceSampler(dset.Y, seed, True)

    if len(sampler) == 0:
        pytest.skip("no indices to sample!")

    assert list(sampler) != list(sampler)


def test_seed_class_balance_reproducibility(smis, bin_targets, featurizer, seed):
    data = [MoleculeDatapoint.from_smi(smi, target) for smi, target in zip(smis, bin_targets)]
    dset = MoleculeDataset(data, featurizer)

    sampler1 = ClassBalanceSampler(dset.Y, seed, True)
    sampler2 = ClassBalanceSampler(dset.Y, seed, True)

    assert list(sampler1) == list(sampler2)
