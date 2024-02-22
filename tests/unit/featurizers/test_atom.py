"""NOTE: these tests make a lot of assumptions about the internal mechanics of the AtomFeaturizer,
so they'll need to be reworked if something ever changes about that."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType

from chemprop.featurizers import MultiHotAtomFeaturizer


SMI = "Cn1nc(CC(=O)Nc2ccc3oc4ccccc4c3c2)c2ccccc2c1=O"


@pytest.fixture(params=list(Chem.MolFromSmiles(SMI).GetAtoms())[:5])
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
        HybridizationType.S,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP2D,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
        HybridizationType.OTHER,
    ]


@pytest.fixture
def featurizer(max_atomic_num, degree, formal_charge, chiral_tag, num_Hs, hybridization):
    return MultiHotAtomFeaturizer(
        max_atomic_num, degree, formal_charge, chiral_tag, num_Hs, hybridization
    )


@pytest.fixture
def expected_len(max_atomic_num, degree, formal_charge, chiral_tag, num_Hs, hybridization):
    return (
        (max_atomic_num + 1)
        + sum(len(xs) + 1 for xs in (degree, formal_charge, chiral_tag, num_Hs, hybridization))
        + 2
    )


@pytest.fixture
def x(featurizer, atom):
    return featurizer(atom)


def test_len(featurizer, expected_len):
    assert len(featurizer) == expected_len


def test_none(featurizer):
    np.testing.assert_array_equal(featurizer(None), np.zeros(len(featurizer)))


def test_atomic_num_bit(atom, x, max_atomic_num):
    n = atom.GetAtomicNum()

    if n > max_atomic_num:
        assert x[max_atomic_num] == 1
    else:
        assert x[n - 1] == 1


def test_aromatic_bit(x, aromatic):
    i = -2
    if aromatic:
        assert x[i] == 1
    else:
        assert x[i] == 0


def test_mass_bit(x, mass_bit):
    assert x[-1] == pytest.approx(mass_bit)


@pytest.mark.parametrize(
    "a,x_v_orig",
    zip(
        list(Chem.MolFromSmiles("Fc1cccc(C2(c3nnc(Cc4cccc5ccccc45)o3)CCOCC2)c1").GetAtoms()),
        # fmt: off
        [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.18998],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0.12011],            
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0.12011],        
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0.12011],        
        ]
        # fmt: on
    ),
)
def test_x_orig(a, x_v_orig):
    f = MultiHotAtomFeaturizer()
    x_v_calc = f(a)
    print(x_v_calc)

    np.testing.assert_array_almost_equal(x_v_calc, x_v_orig)
