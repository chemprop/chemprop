import numpy as np
import pytest

from chemprop.data import MoleculeDatapoint

SMI = "c1ccccc1"


@pytest.fixture(params=range(1, 3))
def targets(request):
    return np.random.rand(request.param)


@pytest.fixture(params=[0.5, 0.9])
def features(request):
    return np.where(np.random.rand(1024) > request.param, 1.0, 0.0)


@pytest.fixture
def features_with_nans(features):
    idxs = np.random.choice(len(features), len(features) // 100, False)
    features[idxs] = np.nan

    return features


def test_num_tasks(targets):
    d = MoleculeDatapoint.from_smi(SMI, y=targets)

    assert d.t == targets.shape[0]


def test_addh(smi, targets):
    d1 = MoleculeDatapoint.from_smi(smi, y=targets)
    d2 = MoleculeDatapoint.from_smi(smi, y=targets, add_h=True)

    assert d1.mol.GetNumAtoms() != d2.mol.GetNumAtoms()


def test_replace_token(smi, targets, features_with_nans):
    if not np.isnan(features_with_nans).any():
        pytest.skip("no `nan`s")

    d = MoleculeDatapoint.from_smi(smi, y=targets, x_d=features_with_nans)

    assert not np.isnan(d.x_d).any()
