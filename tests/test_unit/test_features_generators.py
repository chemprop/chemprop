import uuid

import pytest
from rdkit import Chem

from chemprop.features.features_generators import (rdkit_2d_features_generator, rdkit_2d_normalized_features_generator, morgan_binary_features_generator, morgan_counts_features_generator, get_available_features_generators, register_features_generator)


@pytest.fixture(params=["C", "c1ccccc1", "CCCC", "CC(=O)C"])
def smi(request):
    return request.param


@pytest.fixture
def mol(smi):
    return Chem.MolFromSmiles(smi)


@pytest.fixture(params=range(1, 3))
def n(request):
    return request.param


@pytest.fixture
def multimol_data(smi, n):
    return [[smi] * n] * n


@pytest.fixture(params=[1024 * 2**i for i in range(3)])
def radius(request):
    return request.param


@pytest.fixture(params=[morgan_binary_features_generator, morgan_counts_features_generator])
def morgan_generator(request):
    return request.param


@pytest.fixture(params=[rdkit_2d_features_generator, rdkit_2d_normalized_features_generator])
def rdkit_generator(request):
    return request.param


@pytest.fixture(params=range(1, 4))
def num_bits(request):
    return request.param


@pytest.fixture
def rdkit_features():
    return 200


class TestShape:
    def test_morgan_smi(self, morgan_generator, smi, radius, num_bits):
        features = morgan_generator(smi, radius, num_bits)

        assert features.shape == (num_bits,)

    def test_morgan_mol(self, morgan_generator, mol, radius, num_bits):
        features = morgan_generator(mol, radius, num_bits)

        assert features.shape == (num_bits,)
    
    def test_morgan_multimol(self, morgan_generator, multimol_data, radius, num_bits):
        features = morgan_generator(multimol_data, radius, num_bits)

        assert features.shape == (len(multimol_data), len(multimol_data[0]), num_bits)

    def test_rdkit_smi(self, rdkit_generator, smi, rdkit_features):
        features = rdkit_generator(smi)

        assert features.shape == (rdkit_features,)
    
    def test_rdkit_mol(self, rdkit_generator, mol, rdkit_features):
        features = rdkit_generator(mol)

        assert features.shape == (rdkit_features,)

    def test_rdkit_multimol(self, rdkit_generator, multimol_data, rdkit_features):
        features = rdkit_generator(multimol_data)

        assert features.shape == (len(multimol_data), len(multimol_data[0]), rdkit_features)


@pytest.fixture
def default_fgs():
    return ("morgan", "morgan_count", "rdkit_2d", "rdkit_2d_normalized")

@pytest.fixture(params=[str(uuid.uuid4()) for _ in range(3)])
def name(request):
    return request.param

class TestGeneratorRegistration:
    def test_defaults(self, default_fgs):
        assert set(get_available_features_generators()) == set(default_fgs)

    def test_custom(self, name):
        @register_features_generator(name)
        def custom_generator(mol):
            return mol

        assert (name in get_available_features_generators())
