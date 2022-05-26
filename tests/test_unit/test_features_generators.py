import pytest
from chemprop.features import features_generators
from rdkit import Chem
from typing import Sequence

TEST_SMILES_STRING = "C"
TEST_MOLECULE_OBJECT = Chem.MolFromSmiles(TEST_SMILES_STRING)

TEST_MULTIMOLECULE_LIST = [
    [TEST_SMILES_STRING, TEST_SMILES_STRING],
    [TEST_SMILES_STRING, TEST_SMILES_STRING],
]

MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048

NUM_RDKIT_FEATURES = 200


# Ensure the output dimensionality of each generator is correct
class TestFeatureGenerators:
    class TestOutputDimensionality:
        @pytest.mark.parametrize(
            "mol_data, radius, num_bits",
            [
                (TEST_SMILES_STRING, MORGAN_RADIUS, MORGAN_NUM_BITS),
                (TEST_MOLECULE_OBJECT, MORGAN_RADIUS, MORGAN_NUM_BITS),
                (TEST_MULTIMOLECULE_LIST, MORGAN_RADIUS, MORGAN_NUM_BITS),
            ],
        )
        def test_morgan(self, mol_data, radius, num_bits):
            """Ensure correct output dimensionality for Morgan fingerprint feature generator."""
            features = features_generators.morgan_binary_features_generator(
                mol_data, radius, num_bits
            )

            if isinstance(mol_data, Sequence) and not isinstance(mol_data, str):
                if isinstance(mol_data[0], Sequence) and not isinstance(
                    mol_data[0], str
                ):
                    assert features.shape == (
                        len(mol_data),
                        len(mol_data[0]),
                        MORGAN_NUM_BITS,
                    )
                else:
                    assert features.shape == (len(mol_data), 1, MORGAN_NUM_BITS)
            else:
                assert features.shape == (MORGAN_NUM_BITS,)

        @pytest.mark.parametrize(
            "mol_data, radius, num_bits",
            [
                (TEST_SMILES_STRING, MORGAN_RADIUS, MORGAN_NUM_BITS),
                (TEST_MOLECULE_OBJECT, MORGAN_RADIUS, MORGAN_NUM_BITS),
                (TEST_MULTIMOLECULE_LIST, MORGAN_RADIUS, MORGAN_NUM_BITS),
            ],
        )
        def test_morgan_counts(self, mol_data, radius, num_bits):
            """Ensure correct output dimensionality for counts-based Morgan fingerprint feature generator."""
            features = features_generators.morgan_counts_features_generator(
                mol_data, radius, num_bits
            )

            if isinstance(mol_data, Sequence) and not isinstance(mol_data, str):
                if isinstance(mol_data[0], Sequence) and not isinstance(
                    mol_data[0], str
                ):
                    assert features.shape == (
                        len(mol_data),
                        len(mol_data[0]),
                        MORGAN_NUM_BITS,
                    )
                else:
                    assert features.shape == (len(mol_data), 1, MORGAN_NUM_BITS)
            else:
                assert features.shape == (MORGAN_NUM_BITS,)

        @pytest.mark.parametrize(
            "mol_data",
            [TEST_SMILES_STRING, TEST_MOLECULE_OBJECT, TEST_MULTIMOLECULE_LIST],
        )
        def test_rdkit2d(self, mol_data):
            """Ensure correct output dimensionality for RDKit 2D feature generator."""
            features = features_generators.rdkit_2d_features_generator(mol_data)

            if isinstance(mol_data, Sequence) and not isinstance(mol_data, str):
                if isinstance(mol_data[0], Sequence) and not isinstance(
                    mol_data[0], str
                ):
                    assert features.shape == (
                        len(mol_data),
                        len(mol_data[0]),
                        NUM_RDKIT_FEATURES,
                    )
                else:
                    assert features.shape == (len(mol_data), 1, NUM_RDKIT_FEATURES)
            else:
                assert features.shape == (NUM_RDKIT_FEATURES,)

        @pytest.mark.parametrize(
            "mol_data",
            [TEST_SMILES_STRING, TEST_MOLECULE_OBJECT, TEST_MULTIMOLECULE_LIST],
        )
        def test_rdkit2d_normalized(self, mol_data):
            """Ensure correct output feature dimensionality for RDKit 2D normalized feature generator."""
            features = features_generators.rdkit_2d_normalized_features_generator(
                mol_data
            )

            if isinstance(mol_data, Sequence) and not isinstance(mol_data, str):
                if isinstance(mol_data[0], Sequence) and not isinstance(
                    mol_data[0], str
                ):
                    assert features.shape == (
                        len(mol_data),
                        len(mol_data[0]),
                        NUM_RDKIT_FEATURES,
                    )
                else:
                    assert features.shape == (len(mol_data), 1, NUM_RDKIT_FEATURES)
            else:
                assert features.shape == (NUM_RDKIT_FEATURES,)

    class TestGeneratorRegistration:
        def test_default_generator_registration(self):
            """Ensure all default feature generators are correctly registered."""
            assert features_generators.get_available_features_generators() == [
                "morgan",
                "morgan_count",
                "rdkit_2d",
                "rdkit_2d_normalized",
            ]

        def test_custom_generator_registration(self):
            """Ensure that a custom feature generator is correctly registered during runtime."""

            @features_generators.register_features_generator("custom_generator")
            def generator(mol):
                return mol

            assert (
                "custom_generator"
                in features_generators.get_available_features_generators()
            )
