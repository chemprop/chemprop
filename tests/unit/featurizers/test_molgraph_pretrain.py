import copy

import numpy as np
import pytest
from rdkit import Chem

from chemprop.data.collate import PreBatchMolGraph
from chemprop.featurizers.molgraph import PretrainMoleculeMolGraphFeaturizer


# Fixture for RDKit molecule from SMILES
@pytest.fixture
def mol():
    SMI = "Cn1nc(CC(=O)Nc2ccc3oc4ccccc4c3c2)c2ccccc2c1=O"
    return Chem.MolFromSmiles(SMI)


# Fixture for Featurizer
@pytest.fixture
def featurizer():
    return PretrainMoleculeMolGraphFeaturizer()


# Test case for basic featurization
def test_featurization(mol, featurizer):
    premolgraph = featurizer(mol)
    assert premolgraph is not None


# Test case for masked atom pretraining
def test_masked_atom_pretraining(mol, featurizer):
    premolgraph = featurizer(mol)
    premolgraph_masked = copy.deepcopy(premolgraph)
    premolgraph_masked.masked_atom_pretraining(0.3)
    num_original_edges = premolgraph.E.shape[0]
    num_original_nodes = premolgraph.V.shape[0]
    len_masked_atom = int(num_original_nodes * 0.3)
    assert (
        premolgraph_masked.V.shape[0],
        premolgraph_masked.E.shape[0],
        premolgraph_masked.masked_atom_index.shape[0],
    ) == (num_original_nodes, num_original_edges, len_masked_atom)


# Test case for subgraph deletion
def test_subgraph_deletion(mol, featurizer):
    premolgraph = featurizer(mol)
    premolgraph_subgraph = copy.deepcopy(premolgraph)
    premolgraph_subgraph.subgraph_deletion(3, 0.3)
    # Add assertions based on expected changes to the graph
    number_of_deleted_bonds_1 = len(premolgraph_subgraph.subgraph_masked_bond_indices)
    number_of_deleted_bonds_2 = premolgraph.E.shape[0] - premolgraph_subgraph.E.shape[0]
    masked_atoms_feature_sum = np.sum(
        premolgraph_subgraph.V[premolgraph_subgraph.subgraph_masked_atom_indices]
    )
    assert (number_of_deleted_bonds_1, 0.0) == (number_of_deleted_bonds_2, masked_atoms_feature_sum)


# Test case for bond deletion
def test_bond_deletion(mol, featurizer):
    premolgraph = featurizer(mol)
    premolgraph_bond = copy.deepcopy(premolgraph)
    premolgraph_bond.bond_deletion_complete(0.3)
    # Add assertions based on expected changes to the graph
    num_delete_bonds = int(premolgraph.E.shape[0] // 2 * 0.3) * 2
    num_delete_bonds_2 = premolgraph.E.shape[0] - premolgraph_bond.E.shape[0]
    assert num_delete_bonds == num_delete_bonds_2


# Fixture for SMILES list
@pytest.fixture
def smiles_list():
    return ["Cn1nc(CC(=O)Nc2ccc3oc4ccccc4c3c2)c2ccccc2c1=O", "C", "CC", "CCC", "C1CC1", "C1CCC1"]


# Test case for batched molecule graph preparation
def test_batch_molgraph(smiles_list, featurizer):
    molgraph_list = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        pre_mol = featurizer(mol)
        molgraph_list.append(pre_mol)

    batch = PreBatchMolGraph(molgraph_list)
    original_batch = batch.prepare_batch()

    # Assertions for batch structure
    num_original_nodes = sum([g.V.shape[0] for g in molgraph_list])
    num_original_edges = sum([g.E.shape[0] for g in molgraph_list])

    assert original_batch.V.shape[0] == num_original_nodes
    assert original_batch.E.shape[0] == num_original_edges
    assert original_batch is not None


# Test case for masked batch operation
def test_masked_batch_operation(smiles_list, featurizer):
    molgraph_list = []
    num_masked_atoms_per_graph = []

    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        pre_mol = featurizer(mol)
        molgraph_list.append(pre_mol)
        num_original_nodes = pre_mol.V.shape[0]
        num_masked_atoms_per_graph.append(int(num_original_nodes * 0.3))

    batch = PreBatchMolGraph(molgraph_list)
    mask_batch = copy.deepcopy(batch).apply_mask(0.3)

    # Ensure that the total number of nodes remains the same across the batch
    total_nodes_before = sum([g.V.shape[0] for g in molgraph_list])
    total_masked_atoms = sum(num_masked_atoms_per_graph)

    assert mask_batch.V.shape[0] == total_nodes_before
    assert mask_batch.batch_masked_atom_index.shape[0] == total_masked_atoms
    assert mask_batch is not None


# Test case for bond deletion in batch
def test_bond_deletion_batch(smiles_list, featurizer):
    molgraph_list = []
    total_original_edges = 0
    total_expected_deleted_bonds = 0

    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        pre_mol = featurizer(mol)
        molgraph_list.append(pre_mol)
        num_original_edges = pre_mol.E.shape[0]
        total_original_edges += num_original_edges
        total_expected_deleted_bonds += (
            int(pre_mol.E.shape[0] // 2 * 0.3) * 2
        )  # 30% bonds deleted for each graph

    batch = PreBatchMolGraph(molgraph_list)
    bond_batch = copy.deepcopy(batch).apply_bond_deletion(0.3)

    # Check total edges after bond deletion
    assert bond_batch.E.shape[0] == total_original_edges - total_expected_deleted_bonds
    assert bond_batch is not None


# Test case for subgraph deletion in batch
def test_subgraph_deletion_batch(smiles_list, featurizer):
    molgraph_list = []
    total_original_edges = 0
    total_original_nodes = 0
    total_deleted_subgraph_bonds = 0

    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        pre_mol = featurizer(mol)
        molgraph_list.append(pre_mol)
        num_original_edges = pre_mol.E.shape[0]
        num_original_nodes = pre_mol.V.shape[0]
        total_original_edges += num_original_edges
        total_original_nodes += num_original_nodes
        total_deleted_subgraph_bonds += round(
            num_original_edges * 0.3
        )  # Simulate 30% bond deletion

    batch = PreBatchMolGraph(molgraph_list)
    subgraph_batch = copy.deepcopy(batch).apply_subgraph_deletion(0, 0.3)

    # Check that the number of edges matches the expected result after subgraph deletion
    assert subgraph_batch.E.shape[0] == total_original_edges - total_deleted_subgraph_bonds
    # Ensure the batch is still valid
    assert subgraph_batch is not None
