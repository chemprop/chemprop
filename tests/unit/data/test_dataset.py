import numpy as np
import pytest

from chemprop.v2.data import MoleculeDataset, MoleculeDatapoint
from chemprop.v2.featurizers import MoleculeMolGraphFeaturizer


@pytest.fixture(
    params=[
        [
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
        ],
        [
            "NC(=O)[C@@H]1Cc2ccccc2CN1C(=O)c1ccc2c(c1)Cc1ccccc1-2",
            "Cc1cc(C(=O)NCc2cccc(NC(=O)CCN3CCOCC3)c2)ccc1-n1cnnn1",
            "CNC(=O)c1cccc(CNC(=O)c2ccc3c4c(cccc24)CC3)c1",
            "CN(CC1COc2ccccc2O1)C(=O)NC(Cn1cnc2ccccc21)c1ccc(F)cc1",
            "O=C(C1CCCCC1)N1CC(c2nc(-c3ccc4c(c3)CCC4)no2)C1",
            "O=C(c1ccc(Cl)cc1)N1CCCN(C(=O)c2ccc3[nH]c(=O)[nH]c3c2)CC1",
            "O=C(CN1C(=O)NC2(CCCCC2)C1=O)Nc1ccc2cc[nH]c(=O)c2c1",
            "O=C(c1ccc(O)cc1)C1CCN(C(=O)NC2COc3ccccc32)CC1",
        ],
        [
            "O=C(c1ccc2c(c1)CNC2)N1CCc2c(cccc2C(F)(F)F)C1",
            "Cc1ccc(OCC(=O)N2CCN(CC(=O)N3CCc4sccc4C3c3ccccc3)CC2)cc1",
            "NC(=O)C1CCN(C(=O)CCc2c(-c3ccccc3)[nH]c3ccc(F)cc23)C1",
            "COc1ccnc(N2CCN(C(=O)c3cc4ccccc4c4cccnc34)CC2)n1",
            "O=C(Nc1n[nH]c2ccc(F)cc12)C1(c2ccccc2)CCOCC1",
        ],
        [
            "Cc1cc(NC(=O)CN2CCN(C(=O)C3CCCc4ccccc43)CC2)no1",
            "C[C@@H](NC(=O)NCc1noc2c1CCCC2)c1cccc2ccccc12",
            "O=C(Cc1c[nH]c2ccccc12)N1CCN(C(=O)c2cc3ccccc3[nH]c2=O)CC1",
            "O=C(NCCc1nnc2n1CCCCC2)NC1CCc2ccccc2C1",
        ],
        ["O=C(Cc1ccc2ccccc2c1)NC(CN1CCCC1=O)c1ccccc1"],
    ]
)
def smis(request):
    return request.param


@pytest.fixture
def targets(smis):
    return np.random.rand(len(smis), 1)


@pytest.fixture
def data(smis, targets):
    return [MoleculeDatapoint(smi, t) for smi, t in zip(smis, targets)]


@pytest.fixture
def dataset(data):
    return MoleculeDataset(data, MoleculeMolGraphFeaturizer())


def test_none():
    with pytest.raises(ValueError):
        MoleculeDataset(None, MoleculeMolGraphFeaturizer())


def test_empty():
    """TODO"""
    pass


def test_len(data, dataset):
    assert len(data) == len(dataset)


def test_smis(dataset, smis):
    assert smis == dataset.smiles


def test_targets(dataset, targets):
    np.testing.assert_array_equal(dataset.targets, targets)


def test_set_targets_too_short(dataset):
    with pytest.raises(ValueError):
        dataset.targets = np.random.rand(len(dataset) // 2, 1)


def test_num_tasks(dataset, targets):
    assert dataset.num_tasks == targets.shape[1]


def test_aux_nones(dataset: MoleculeDataset):
    assert dataset.X_f is None
    assert dataset.X_phase is None
    assert dataset.V_fs is None
    assert dataset.E_fs is None
    assert dataset.gt_mask is None
    assert dataset.lt_mask is None
    assert dataset.d_v is None
    assert dataset.d_vf is None
    assert dataset.d_ef is None
