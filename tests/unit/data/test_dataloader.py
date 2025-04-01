import numpy as np
import pytest
import torch

from chemprop.data.collate import BatchMolGraph, collate_batch, collate_mol_atom_bond_batch
from chemprop.data.datasets import Datum, MolAtomBondDatum
from chemprop.data.molgraph import MolGraph


@pytest.fixture
def datum_1():
    mol_graph1 = MolGraph(
        V=np.array([[1.0], [2.0], [3.0]]),
        E=np.array([[0.5], [1.5]]),
        edge_index=np.array([[0, 1, 0, 2], [1, 0, 2, 0]]),
        rev_edge_index=np.array([1, 0, 3, 2]),
    )
    return Datum(
        mol_graph1,
        V_d=np.array([[1.0], [2.0], [4.0]]),
        x_d=[3, 4],
        y=[6, 7],
        weight=[8.0],
        lt_mask=[True],
        gt_mask=[False],
    )


@pytest.fixture
def datum_2():
    mol_graph2 = MolGraph(
        V=np.array([[4.0], [5.0]]),
        E=np.array([[2.5]]),
        edge_index=np.array([[0, 1], [1, 0]]),
        rev_edge_index=np.array([1, 0]),
    )
    return Datum(
        mol_graph2,
        V_d=np.array([[5.0], [7.0]]),
        x_d=[8, 9],
        y=[6, 4],
        weight=[1.0],
        lt_mask=[False],
        gt_mask=[True],
    )


def test_collate_batch_single_graph(datum_1):
    batch = [datum_1]

    result = collate_batch(batch)
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = result

    assert isinstance(result, tuple)
    assert isinstance(mgs, BatchMolGraph)
    assert (
        mgs.V.shape[0] == V_ds.shape[0]
    )  # V is number of atoms x number of atom features, V_ds is number of atoms x number of atom descriptors
    torch.testing.assert_close(V_ds, torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float32))
    torch.testing.assert_close(x_ds, torch.tensor([[3.0, 4.0]], dtype=torch.float32))
    torch.testing.assert_close(ys, torch.tensor([[6.0, 7.0]], dtype=torch.float32))
    torch.testing.assert_close(weights, torch.tensor([[[8.0]]], dtype=torch.float32))
    torch.testing.assert_close(lt_masks, torch.tensor([[1]], dtype=torch.bool))
    torch.testing.assert_close(gt_masks, torch.tensor([[0]], dtype=torch.bool))


def test_collate_batch_multiple_graphs(datum_1, datum_2):
    batch = [datum_1, datum_2]

    result = collate_batch(batch)
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = result

    assert isinstance(mgs, BatchMolGraph)
    assert (
        mgs.V.shape[0] == V_ds.shape[0]
    )  # V is number of atoms x number of atom features, V_ds is number of atoms x number of atom descriptors
    torch.testing.assert_close(
        V_ds, torch.tensor([[1.0], [2.0], [4.0], [5.0], [7.0]], dtype=torch.float32)
    )
    torch.testing.assert_close(x_ds, torch.tensor([[3.0, 4.0], [8.0, 9.0]], dtype=torch.float32))
    torch.testing.assert_close(ys, torch.tensor([[6.0, 7.0], [6.0, 4.0]], dtype=torch.float32))
    torch.testing.assert_close(weights, torch.tensor([[[8.0]], [[1.0]]], dtype=torch.float32))
    torch.testing.assert_close(lt_masks, torch.tensor([[1], [0]], dtype=torch.bool))
    torch.testing.assert_close(gt_masks, torch.tensor([[0], [1]], dtype=torch.bool))


@pytest.fixture
def mol_atom_bond_datum_1():
    mol_graph1 = MolGraph(
        V=np.array([[1.0], [2.0], [3.0]]),
        E=np.array([[0.5], [1.5]]),
        edge_index=np.array([[0, 1, 0, 2], [1, 0, 2, 0]]),
        rev_edge_index=np.array([1, 0, 3, 2]),
    )
    return MolAtomBondDatum(
        mol_graph1,
        V_d=np.array([[1.0], [2.0], [4.0]]),
        E_d=None,
        x_d=[3, 4],
        ys=[
            np.array([12]).reshape(1, 1),
            np.array([6, 7]).reshape(2, 1),
            np.array([3, 4, 5]).reshape(3, 1),
        ],
        weight=8.0,
        lt_masks=[True, True, True],
        gt_masks=[False, False, False],
    )


@pytest.fixture
def mol_atom_bond_datum_2():
    mol_graph2 = MolGraph(
        V=np.array([[4.0], [5.0]]),
        E=np.array([[2.5]]),
        edge_index=np.array([[0, 1], [1, 0]]),
        rev_edge_index=np.array([1, 0]),
    )
    return MolAtomBondDatum(
        mol_graph2,
        V_d=np.array([[5.0], [7.0]]),
        E_d=None,
        x_d=[8, 9],
        ys=[
            np.array(np.nan).reshape(1, 1),
            np.array([6, 4]).reshape(2, 1),
            np.array([np.nan, np.nan, np.nan]).reshape(3, 1),
        ],
        weight=1.0,
        lt_masks=[False, False, False],
        gt_masks=[True, True, True],
    )


def test_collate_mol_atom_bond_batch_single_graph(mol_atom_bond_datum_1):
    batch = [mol_atom_bond_datum_1]

    result = collate_mol_atom_bond_batch(batch)
    mgs, V_ds, E_ds, x_ds, ys, weights, lt_masks, gt_masks = result

    assert isinstance(result, tuple)
    assert isinstance(mgs, BatchMolGraph)
    assert (
        mgs.V.shape[0] == V_ds.shape[0]
    )  # V is number of atoms x number of atom features, V_ds is number of atoms x number of atom descriptors
    torch.testing.assert_close(V_ds, torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float32))
    torch.testing.assert_close(x_ds, torch.tensor([[3.0, 4.0]], dtype=torch.float32))
    torch.testing.assert_close(ys[0], torch.tensor([[12.0]], dtype=torch.float32))
    torch.testing.assert_close(ys[1], torch.tensor([[6.0, 7.0]], dtype=torch.float32).reshape(2, 1))
    torch.testing.assert_close(
        ys[2], torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32).reshape(3, 1)
    )
    torch.testing.assert_close(weights[0], torch.tensor([8.0], dtype=torch.float32))
    torch.testing.assert_close(lt_masks[0], torch.tensor([[1]], dtype=torch.bool))
    torch.testing.assert_close(gt_masks[0], torch.tensor([[0]], dtype=torch.bool))


def test_collate_mol_atom_bond_batch_multiple_graphs(mol_atom_bond_datum_1, mol_atom_bond_datum_2):
    batch = [mol_atom_bond_datum_1, mol_atom_bond_datum_2]

    result = collate_mol_atom_bond_batch(batch)
    mgs, V_ds, E_ds, x_ds, ys, weights, lt_masks, gt_masks = result

    assert isinstance(mgs, BatchMolGraph)
    assert (
        mgs.V.shape[0] == V_ds.shape[0]
    )  # V is number of atoms x number of atom features, V_ds is number of atoms x number of atom descriptors
    torch.testing.assert_close(
        V_ds, torch.tensor([[1.0], [2.0], [4.0], [5.0], [7.0]], dtype=torch.float32)
    )
    torch.testing.assert_close(x_ds, torch.tensor([[3.0, 4.0], [8.0, 9.0]], dtype=torch.float32))
    torch.testing.assert_close(
        ys[1], torch.tensor([[6.0, 7.0, 6.0, 4.0]], dtype=torch.float32).reshape(4, 1)
    )
    torch.testing.assert_close(weights[0], torch.tensor([8.0, 1.0], dtype=torch.float32))
    torch.testing.assert_close(lt_masks[0], torch.tensor([[1], [0]], dtype=torch.bool))
    torch.testing.assert_close(gt_masks[0], torch.tensor([[0], [1]], dtype=torch.bool))
