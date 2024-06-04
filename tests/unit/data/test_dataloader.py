import numpy as np
import pytest
import torch

from chemprop.data.collate import BatchMolGraph, collate_batch
from chemprop.data.datasets import Datum
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
