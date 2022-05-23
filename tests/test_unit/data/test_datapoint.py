import numpy as np
import pytest

from chemprop.data.v2.molecule import MoleculeDatapoint
from chemprop.featurizers.features_generators import get_available_features_generators

@pytest.fixture(params=[
    'OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O',
    'Cc1occc1C(=O)Nc2ccccc2',
    'CC(C)=CCCC(C)=CC(=O)',
    'c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43',
    'c1ccsc1',
    'c2ccc1scnc1c2'
])
def smi(request):
    return request.param


@pytest.fixture(params=range(1, 3))
def targets(request):
    return np.random.rand(request.param)


@pytest.fixture(params=[0.5, 0.9, 0.99])
def features(request):
    return np.where(np.random.rand(1024) > request.param, 1.0, 0.0)


@pytest.fixture
def features_with_nans(features):
    idxs = np.random.choice(len(features), len(features) // 100, False)
    features[idxs] = np.nan

    return features


@pytest.fixture
def features_generators():
    return get_available_features_generators()


@pytest.fixture
def datapoint(smi, targets):
    return MoleculeDatapoint(smi, targets)


def test_features_and_fg(features_generators):
    smi = "c1ccccc1"
    targets = np.random.rand(1, 1)
    features = np.random.rand(1024)
    with pytest.raises(ValueError):
        MoleculeDatapoint(smi, targets, features=features, features_generators=features_generators)


def test_num_tasks(smi, targets):
    d = MoleculeDatapoint(smi, targets)

    assert d.num_tasks == targets.shape[0]


def test_addh(smi, targets):
    d1 = MoleculeDatapoint(smi, targets)
    d2 = MoleculeDatapoint(smi, targets, add_h=True)

    assert d1.mol.GetNumAtoms() != d2.mol.GetNumAtoms


def test_replace_token(smi, targets, features_with_nans):
    if not np.isnan(features_with_nans).any():
        pytest.skip("no `nan`s")
    
    d = MoleculeDatapoint(smi, targets, features=features_with_nans)

    assert not np.isnan(d.features).any()
    