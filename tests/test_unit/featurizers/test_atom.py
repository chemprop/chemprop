
from rdkit.Chem.rdchem import Atom
import pytest

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

@pytest.fixture(params=["B", "C", "N", "O", "Cl"])
def atom(request):
    return Atom(request.param)


@pytest.fixture(params=[10, 50, 100])
def max_atomic_num(request):
    return request.param


@pytest.fixture
def featurizer(max_atomic_num):
    pass


def test_len(featurizer):
    pass


def test_none(featurizer, atom):
    pass


def test_atomic_num_bit(max_atomic_num, featurizer, atom):
    pass