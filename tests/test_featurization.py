import pytest
from rdkit.Chem import rdchem

from chemprop.features import featurization

@pytest.fixture(params=[0, 10, 50, 100])
def max_atomic_num(request):
    return request.param

@pytest.fixture(params=range(10))
def atom(request):
    return rdchem.Atom(request.param + 1)

def test_atom_feat_params_atomic_num(max_atomic_num: int):
    params = featurization.AtomFeaturizationParams(max_atomic_num)

    assert params.atomic_num == list(range(max_atomic_num))

def test_feat_params_max_atomic_num(max_atomic_num: int):
    params = featurization.FeaturizationParams(max_atomic_num)

    assert params.atom_features.atomic_num == list(range(max_atomic_num))

def test_feat_params_set_max_atomic_num(max_atomic_num: int):
    params = featurization.FeaturizationParams()
    params.max_atomic_num = max_atomic_num

    assert params.max_atomic_num == max_atomic_num
    assert params.atom_features.atomic_num == list(range(params.max_atomic_num))
    assert params.atom_fdim == len(params.atom_features)

def test_feat_params_atom_fdim(max_atomic_num: int):
    params = featurization.FeaturizationParams(max_atomic_num)

    assert params.atom_fdim == len(params.atom_features)

@pytest.mark.parametrize("x", [10, 20, 40])
def test_reset_featurization(x: int):
    featurization.PARAMS.max_atomic_num += x

    old_max_atomic_num = featurization.PARAMS.max_atomic_num
    featurization.reset_featurization_parameters()

    new_max_atomic_num = featurization.PARAMS.max_atomic_num

    assert old_max_atomic_num != new_max_atomic_num

def test_atom_features_None():
    features = featurization.atom_features(None)

    assert not any(features)

def test_atom_features_zeros(atom):
    features = featurization.atom_features_zeros(atom)
    idx = atom.GetAtomicNum() - 1

    assert not any(features[:idx]) and features[idx] and not any(features[idx+1:])

def test_bond_features_None():
    features = featurization.bond_features(None)

    assert features[0] and not any(features[1:])