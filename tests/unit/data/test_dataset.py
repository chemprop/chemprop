import numpy as np
import pytest

from chemprop.data import MoleculeDataset, MoleculeDatapoint
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer


@pytest.fixture(params=[1, 5, 10])
def smis(smis, request):
    return smis.sample(request.param).to_list()


@pytest.fixture
def targets(smis):
    return np.random.rand(len(smis), 1)


@pytest.fixture
def data(smis, targets):
    return [MoleculeDatapoint.from_smi(smi, t) for smi, t in zip(smis, targets)]


@pytest.fixture
def dataset(data):
    return MoleculeDataset(data, SimpleMoleculeMolGraphFeaturizer())


def test_none():
    with pytest.raises(ValueError):
        MoleculeDataset(None, SimpleMoleculeMolGraphFeaturizer())


def test_empty():
    """TODO"""


def test_len(data, dataset):
    assert len(data) == len(dataset)


def test_smis(dataset, smis):
    assert smis == dataset.smiles


def test_targets(dataset, targets):
    np.testing.assert_array_equal(dataset.Y, targets)


def test_set_targets_too_short(dataset):
    with pytest.raises(ValueError):
        dataset.Y = np.random.rand(len(dataset) // 2, 1)


def test_num_tasks(dataset, targets):
    assert dataset.t == targets.shape[1]


def test_aux_nones(dataset: MoleculeDataset):
    np.testing.assert_array_equal(dataset.X_f, None)
    np.testing.assert_array_equal(dataset.X_f, None)
    np.testing.assert_array_equal(dataset.V_fs, None)
    np.testing.assert_array_equal(dataset.E_fs, None)
    np.testing.assert_array_equal(dataset.gt_mask, None)
    np.testing.assert_array_equal(dataset.lt_mask, None)
    assert dataset.d_vd == 0
    assert dataset.d_vf == 0
    assert dataset.d_ef == 0
