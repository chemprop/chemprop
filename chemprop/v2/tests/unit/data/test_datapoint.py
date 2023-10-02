import numpy as np
import pytest

from chemprop.v2.data import MoleculeDatapoint
#from chemprop.featurizers.features_generators import get_available_features_generators
from chemprop.v2.featurizers.featurizers import MoleculeFeaturizerRegistry

@pytest.fixture(
    params=[
        "Fc1cccc(C2(c3nnc(Cc4cccc5ccccc45)o3)CCOCC2)c1",
        "O=C(NCc1ccnc(Oc2ccc(F)cc2)c1)c1[nH]nc2c1CCCC2",
        "Cc1ccccc1CC(=O)N1CCN(CC(=O)N2Cc3ccccc3C(c3ccccc3)C2)CC1",
        "O=C(Nc1cc2c(cn1)CCCC2)N1CCCC1c1ccc(O)cc1",
        "NC(=O)C1CCN(C(=O)CCc2c(-c3ccc(F)cc3)[nH]c3ccccc23)C1",
        "O=C(NC1CCCCNC1=O)c1cc2ccccc2c2cccnc12",
        "O=C(Cc1ccc(-n2cccn2)cc1)N1CCC(c2nnc3ccccn23)CC1",
        "Cn1nc(CC(=O)Nc2ccc3oc4ccccc4c3c2)c2ccccc2c1=O",
        "O=C(NCC(c1ccccc1)c1ccccc1)N1CCc2ccc(O)cc2C1",
        "O=C(NCc1ccc(Cn2ccccc2=O)cc1)c1ccccc1CCc1ccccc1",
        "NC(=O)[C@@H]1Cc2ccccc2CN1C(=O)c1ccc2c(c1)Cc1ccccc1-2",
        "Cc1cc(C(=O)NCc2cccc(NC(=O)CCN3CCOCC3)c2)ccc1-n1cnnn1",
        "CNC(=O)c1cccc(CNC(=O)c2ccc3c4c(cccc24)CC3)c1",
        "CN(CC1COc2ccccc2O1)C(=O)NC(Cn1cnc2ccccc21)c1ccc(F)cc1",
        "O=C(C1CCCCC1)N1CC(c2nc(-c3ccc4c(c3)CCC4)no2)C1",
        "O=C(c1ccc(Cl)cc1)N1CCCN(C(=O)c2ccc3[nH]c(=O)[nH]c3c2)CC1",
        "O=C(CN1C(=O)NC2(CCCCC2)C1=O)Nc1ccc2cc[nH]c(=O)c2c1",
        "O=C(c1ccc(O)cc1)C1CCN(C(=O)NC2COc3ccccc32)CC1",
        "NC(=O)c1cccc(CNS(=O)(=O)c2ccc3c4c(cccc24)CC3)c1",
    ]
)
def smi(request):
    return request.param


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


@pytest.fixture
def features_generators():
    return MoleculeFeaturizerRegistry.keys()


def test_features_and_fg(features_generators):
    smi = "c1ccccc1"
    targets = np.random.rand(1, 1)
    features = np.random.rand(1024)
    with pytest.raises(ValueError):
        MoleculeDatapoint(smi, targets, features=features, molecule_featurizers=features_generators)


def test_num_tasks(targets):
    d = MoleculeDatapoint("c1ccccc1", targets)

    assert d.t == targets.shape[0]


def test_addh(smi, targets):
    d1 = MoleculeDatapoint(smi, targets)
    d2 = MoleculeDatapoint(smi, targets, add_h=True)

    assert d1.mol.GetNumAtoms() != d2.mol.GetNumAtoms


def test_replace_token(smi, targets, features_with_nans):
    if not np.isnan(features_with_nans).any():
        pytest.skip("no `nan`s")

    d = MoleculeDatapoint(smi, targets, features=features_with_nans)

    assert not np.isnan(d.x_f).any()
