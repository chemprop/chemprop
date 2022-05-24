
"""NOTE: these tests make a lot of assumptions about the internal mechanics of the AtomFeaturizer,
so they'll need to be reworked if something ever changes about that."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType

from chemprop.featurizers.multihot.atom import AtomFeaturizer

smis = [
    'Fc1cccc(C2(c3nnc(Cc4cccc5ccccc45)o3)CCOCC2)c1',
    'O=C(NCc1ccnc(Oc2ccc(F)cc2)c1)c1[nH]nc2c1CCCC2',
    'Cc1ccccc1CC(=O)N1CCN(CC(=O)N2Cc3ccccc3C(c3ccccc3)C2)CC1',
    'O=C(Nc1cc2c(cn1)CCCC2)N1CCCC1c1ccc(O)cc1',
    'NC(=O)C1CCN(C(=O)CCc2c(-c3ccc(F)cc3)[nH]c3ccccc23)C1',
    'O=C(NC1CCCCNC1=O)c1cc2ccccc2c2cccnc12',
    'O=C(Cc1ccc(-n2cccn2)cc1)N1CCC(c2nnc3ccccn23)CC1',
    'Cn1nc(CC(=O)Nc2ccc3oc4ccccc4c3c2)c2ccccc2c1=O',
    'O=C(NCC(c1ccccc1)c1ccccc1)N1CCc2ccc(O)cc2C1',
    'O=C(NCc1ccc(Cn2ccccc2=O)cc1)c1ccccc1CCc1ccccc1'
]

@pytest.fixture(
    params=list(
        Chem.MolFromSmiles("Cn1nc(CC(=O)Nc2ccc3oc4ccccc4c3c2)c2ccccc2c1=O").GetAtoms()
    )[:10]
)
def atom(request):
    return request.param


@pytest.fixture
def aromatic(atom):
    return atom.GetIsAromatic()


@pytest.fixture
def mass_bit(atom):
    return 0.01 * atom.GetMass()


@pytest.fixture(params=[0, 10, 100])
def max_atomic_num(request):
    return request.param


@pytest.fixture
def degree():
    return list(range(6))

@pytest.fixture
def formal_charge():
    return [-1, -2, 1, 2, 0]


@pytest.fixture
def chiral_tag():
    return list(range(4))


@pytest.fixture
def num_Hs():
    return list(range(5))


@pytest.fixture
def hybridization():
    return [
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
    ]


@pytest.fixture
def featurizer(max_atomic_num, degree, formal_charge, chiral_tag, num_Hs, hybridization):
    return AtomFeaturizer(max_atomic_num, degree, formal_charge, chiral_tag, num_Hs, hybridization)


@pytest.fixture
def expected_len(max_atomic_num, degree, formal_charge, chiral_tag, num_Hs, hybridization):
    return (max_atomic_num + 1) + sum(
        len(xs) + 1 for xs in (degree, formal_charge, chiral_tag, num_Hs, hybridization)
    ) + 2


@pytest.fixture
def x(featurizer, atom):
    return featurizer(atom)


def test_len(featurizer, expected_len):
    assert len(featurizer) == expected_len


def test_num_subfeatures():
    assert AtomFeaturizer().num_subfeatures == 8


def test_none(featurizer):
    np.testing.assert_array_equal(
        featurizer(None),
        np.zeros(len(featurizer))
    )


def test_atomic_num_bit(atom, x, max_atomic_num):
    n = atom.GetAtomicNum()

    if n > max_atomic_num:
        assert x[max_atomic_num] == 1
    else:
        assert x[n - 1] == 1


def test_aromatic_bit(featurizer, x, aromatic):
    i = featurizer.subfeatures["aromatic"]
    if aromatic:
        assert x[i] == 1
    else:
        assert x[i] == 0


def test_mass_bit(featurizer, x, mass_bit):
    assert x[featurizer.subfeatures["mass"]] == pytest.approx(mass_bit)