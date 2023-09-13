import pytest
import torch
import numpy as np

from chemprop.v2.data.dataloader import collate_batch
from chemprop.v2.data.datasets import Datum
from chemprop.v2.featurizers.molgraph import MolGraph, BatchMolGraph


mol_graph1 = MolGraph(
    n_atoms=3,
    n_bonds=2,
    V=np.array([[1.0], [2.0], [3.0]]),
    E=np.array([[0.5], [1.5]]),
    a2b=[(0,), (1,), (1,)],
    b2a=[0, 2],
    b2revb=np.array([1, 0]),
    a2a=[0, 1, 2],
    b2b=np.array([0, 1])
)

mol_graph2 = MolGraph(
    n_atoms=2,
    n_bonds=1,
    V=np.array([[4.0], [5.0]]),
    E=np.array([[2.5]]),
    a2b=[(0,), (0,)],
    b2a=[1],
    b2revb=np.array([0]),
    a2a=None,
    b2b=None
)

datum_1 = Datum(
    mol_graph1,
    V_d = np.array([1.0, 2.0]), 
    x_f = [3, 4], 
    y = [6, 7], 
    weight = [8], 
    lt_mask = [9], 
    gt_mask= [10]
)

datum_2 = Datum(
    mol_graph2,
    V_d = np.array([5.0, 7.0]), 
    x_f = [8, 9], 
    y = [6, 4], 
    weight = [1], 
    lt_mask = [2], 
    gt_mask= [3]
)


def test_collate_batch_single_graph():
    
    batch = [
        datum_1,
    ]
    
    result = collate_batch(batch)
    mgs, V_ds, x_fs, ys, weights, lt_masks, gt_masks = result
    
    assert isinstance(result, tuple)
    assert isinstance(mgs, BatchMolGraph)
    assert torch.equal(V_ds, torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert torch.equal(x_fs, torch.tensor([[3.0, 4.0]], dtype=torch.float32))
    assert torch.equal(ys, torch.tensor([[6.0, 7.0]], dtype=torch.float32))
    assert torch.equal(weights, torch.tensor([[[8.0]]], dtype=torch.float32))
    assert torch.equal(lt_masks, torch.tensor([[9.0]], dtype=torch.float32))
    assert torch.equal(gt_masks, torch.tensor([[10.0]], dtype=torch.float32))


def test_collate_batch_multiple_graphs():
    
    batch = [datum_1, datum_2]
    
    result = collate_batch(batch)
    mgs, V_ds, x_fs, ys, weights, lt_masks, gt_masks = result
    
    assert isinstance(result, tuple)
    assert isinstance(mgs, BatchMolGraph)
    assert torch.equal(V_ds, torch.tensor([1.0, 2.0, 5.0, 7.0], dtype=torch.float32))
    assert torch.equal(x_fs, torch.tensor([[3.0, 4.0], [8.0, 9.0]], dtype=torch.float32))
    assert torch.equal(ys, torch.tensor([[6.0, 7.0], [6.0, 4.0]], dtype=torch.float32))
    assert torch.equal(weights, torch.tensor([[[8.0]], [[1.0]]], dtype=torch.float32))
    assert torch.equal(lt_masks, torch.tensor([[9.0], [2.0]], dtype=torch.float32)) 
    assert torch.equal(gt_masks, torch.tensor([[10.0], [3.0]], dtype=torch.float32)) 