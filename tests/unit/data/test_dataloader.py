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
        x_d=np.array([3, 4]),
        y=np.array([6, 7]),
        weight=8.0,
        lt_mask=np.array([True, False]),
        gt_mask=np.array([False, True]),
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
        x_d=np.array([8, 9]),
        y=np.array([6, 4]),
        weight=1.0,
        lt_mask=np.array([False, True]),
        gt_mask=np.array([True, False]),
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
    torch.testing.assert_close(weights, torch.tensor([[8.0]], dtype=torch.float32))
    torch.testing.assert_close(lt_masks, torch.tensor([[1, 0]], dtype=torch.bool))
    torch.testing.assert_close(gt_masks, torch.tensor([[0, 1]], dtype=torch.bool))


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
    torch.testing.assert_close(weights, torch.tensor([[8.0], [1.0]], dtype=torch.float32))
    torch.testing.assert_close(lt_masks, torch.tensor([[1, 0], [0, 1]], dtype=torch.bool))
    torch.testing.assert_close(gt_masks, torch.tensor([[0, 1], [1, 0]], dtype=torch.bool))


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
        E_d=np.array([[0.7], [2.8]]),
        x_d=np.array([3, 4]),
        ys=[
            np.array([12]).reshape(1, 1),
            np.array([6, 7, 8]).reshape(3, 1),
            np.array([3, 4]).reshape(2, 1),
        ],
        weight=8.0,
        lt_masks=[
            np.array([True]).reshape(1, 1),
            np.array([True, False, True]).reshape(3, 1),
            np.array([True, False]).reshape(2, 1),
        ],
        gt_masks=[
            np.array([False]).reshape(1, 1),
            np.array([False, True, False]).reshape(3, 1),
            np.array([False, False]).reshape(2, 1),
        ],
        constraints=[np.array([1.0]), None],
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
        E_d=np.array([[1.0]]),
        x_d=np.array([8, 9]),
        ys=[
            np.array(np.nan).reshape(1, 1),
            np.array([6, np.nan]).reshape(2, 1),
            np.array([0.1]).reshape(1, 1),
        ],
        weight=1.0,
        lt_masks=[
            np.array([False]).reshape(1, 1),
            np.array([False, False]).reshape(2, 1),
            np.array([True]).reshape(1, 1),
        ],
        gt_masks=[
            np.array([True]).reshape(1, 1),
            np.array([True, False]).reshape(2, 1),
            np.array([False]).reshape(1, 1),
        ],
        constraints=[np.array([2.0]), None],
    )


def test_collate_mol_atom_bond_batch_single_graph(mol_atom_bond_datum_1):
    batch = [mol_atom_bond_datum_1]

    result = collate_mol_atom_bond_batch(batch)
    mgs, V_ds, E_ds, x_ds, ys, weights, lt_masks, gt_masks, constraints = result

    assert isinstance(result, tuple)
    assert isinstance(mgs, BatchMolGraph)
    assert mgs.V.shape[0] == V_ds.shape[0]
    torch.testing.assert_close(V_ds, torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float32))
    torch.testing.assert_close(E_ds, torch.tensor([[0.7], [2.8]], dtype=torch.float32))
    torch.testing.assert_close(x_ds, torch.tensor([[3.0, 4.0]], dtype=torch.float32))
    torch.testing.assert_close(ys[0], torch.tensor([[12.0]], dtype=torch.float32))
    torch.testing.assert_close(
        ys[1], torch.tensor([[6.0, 7.0, 8.0]], dtype=torch.float32).reshape(3, 1)
    )
    torch.testing.assert_close(ys[2], torch.tensor([[3.0, 4.0]], dtype=torch.float32).reshape(2, 1))
    torch.testing.assert_close(weights[0], torch.tensor([[8.0]], dtype=torch.float32))
    torch.testing.assert_close(weights[1], torch.tensor([[8.0], [8.0], [8.0]], dtype=torch.float32))
    torch.testing.assert_close(weights[2], torch.tensor([[8.0], [8.0]], dtype=torch.float32))
    torch.testing.assert_close(lt_masks[0], torch.tensor([[1]], dtype=torch.bool))
    torch.testing.assert_close(lt_masks[1], torch.tensor([[1], [0], [1]], dtype=torch.bool))
    torch.testing.assert_close(lt_masks[2], torch.tensor([[1], [0]], dtype=torch.bool))
    torch.testing.assert_close(gt_masks[0], torch.tensor([[0]], dtype=torch.bool))
    torch.testing.assert_close(gt_masks[1], torch.tensor([[0], [1], [0]], dtype=torch.bool))
    torch.testing.assert_close(gt_masks[2], torch.tensor([[0], [0]], dtype=torch.bool))


def test_collate_mol_atom_bond_batch_multiple_graphs(mol_atom_bond_datum_1, mol_atom_bond_datum_2):
    batch = [mol_atom_bond_datum_1, mol_atom_bond_datum_2]

    result = collate_mol_atom_bond_batch(batch)
    mgs, V_ds, E_ds, x_ds, ys, weights, lt_masks, gt_masks, constraints = result

    assert isinstance(mgs, BatchMolGraph)
    assert mgs.V.shape[0] == V_ds.shape[0]
    torch.testing.assert_close(
        V_ds, torch.tensor([[1.0], [2.0], [4.0], [5.0], [7.0]], dtype=torch.float32)
    )
    torch.testing.assert_close(E_ds, torch.tensor([[0.7], [2.8], [1.0]], dtype=torch.float32))
    torch.testing.assert_close(x_ds, torch.tensor([[3.0, 4.0], [8.0, 9.0]], dtype=torch.float32))
    torch.testing.assert_close(
        ys[1],
        torch.tensor([[6.0, 7.0, 8.0, 6.0, np.nan]], dtype=torch.float32).reshape(5, 1),
        equal_nan=True,
    )
    torch.testing.assert_close(weights[0], torch.tensor([[8.0], [1.0]], dtype=torch.float32))
    torch.testing.assert_close(lt_masks[0], torch.tensor([[1], [0]], dtype=torch.bool))
    torch.testing.assert_close(
        gt_masks[1], torch.tensor([[0], [1], [0], [1], [0]], dtype=torch.bool)
    )
    torch.testing.assert_close(constraints[0], torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    assert constraints[1] is None


@pytest.fixture
def mol_atom_bond_datum_no_mol_y():
    mol_graph1 = MolGraph(
        V=np.array([[1.0], [2.0], [3.0]]),
        E=np.array([[0.5], [1.5]]),
        edge_index=np.array([[0, 1, 0, 2], [1, 0, 2, 0]]),
        rev_edge_index=np.array([1, 0, 3, 2]),
    )
    return MolAtomBondDatum(
        mol_graph1,
        V_d=np.array([[1.0], [2.0], [4.0]]),
        E_d=np.array([[0.7], [2.8]]),
        x_d=np.array([3, 4]),
        ys=[None, np.array([6, 7, 8]).reshape(3, 1), np.array([3, 4]).reshape(2, 1)],
        weight=8.0,
        lt_masks=[None, None, None],
        gt_masks=[None, None, None],
        constraints=[None, None],
    )


def test_collate_mol_atom_bond_no_mol_y(mol_atom_bond_datum_no_mol_y):
    batch = [mol_atom_bond_datum_no_mol_y]

    result = collate_mol_atom_bond_batch(batch)
    mgs, V_ds, E_ds, x_ds, ys, weights, lt_masks, gt_masks, constraints = result

    assert ys[0] is None
    assert all([lt_mask is None for lt_mask in lt_masks])


@pytest.fixture
def mol_atom_bond_datum_no_atom_bond_y():
    mol_graph1 = MolGraph(
        V=np.array([[1.0], [2.0], [3.0]]),
        E=np.array([[0.5], [1.5]]),
        edge_index=np.array([[0, 1, 0, 2], [1, 0, 2, 0]]),
        rev_edge_index=np.array([1, 0, 3, 2]),
    )
    return MolAtomBondDatum(
        mol_graph1,
        V_d=np.array([[1.0], [2.0], [4.0]]),
        E_d=np.array([[0.7], [2.8]]),
        x_d=np.array([3, 4]),
        ys=[np.array([12]).reshape(1, 1), None, None],
        weight=8.0,
        lt_masks=[None, None, None],
        gt_masks=[None, None, None],
        constraints=[None, None],
    )


def test_collate_mol_atom_bond_no_atom_bond_y(mol_atom_bond_datum_no_atom_bond_y):
    batch = [mol_atom_bond_datum_no_atom_bond_y]

    result = collate_mol_atom_bond_batch(batch)
    mgs, V_ds, E_ds, x_ds, ys, weights, lt_masks, gt_masks, constraints = result

    assert ys[1] is None
    assert ys[2] is None
    assert constraints is None
