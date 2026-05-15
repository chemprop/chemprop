import torch
from torch import nn
import pytest
import numpy as np
import numpy as np
from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.nn.message_passing import MultiweightMessagePassing, BondMessagePassing, AtomMessagePassing
from chemprop.nn.agg import MeanAggregation
from chemprop.data.molgraph import MolGraph
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM

@pytest.fixture
def dummy_bmg():
    # Create a very simple graph: 2 atoms, 1 bond
    # Use the correct dimensions from chemprop.conf
    V = np.random.rand(2, DEFAULT_ATOM_FDIM).astype(np.float32)
    E = np.random.rand(2, DEFAULT_BOND_FDIM).astype(np.float32)
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    rev_edge_index = np.array([1, 0], dtype=np.int64)

    mg = MolGraph(V=V, E=E, edge_index=edge_index, rev_edge_index=rev_edge_index)
    return BatchMolGraph([mg])

def test_multiweight_message_passing_init():
    depth = 3
    mp = MultiweightMessagePassing(depth=depth)

    # The number of weight matrices should be depth - 1
    assert len(mp.W_h) == depth - 1
    # The number of layer norms should be depth - 1
    assert len(mp.norms) == depth - 1

    # Check that each W_h is a Linear layer
    for layer in mp.W_h:
        assert isinstance(layer, nn.Linear)

def test_multiweight_message_passing_forward(dummy_bmg):
    depth = 3
    mp = MultiweightMessagePassing(depth=depth)
    mp.to("cpu")

    # Forward pass should work
    output = mp(dummy_bmg)

    # Output should be a tensor of shape (num_nodes, d_h)
    # In our dummy graph, num_nodes = 2
    assert output.shape[0] == 2
    assert output.shape[1] == mp.W_h[0].out_features

def test_multiweight_message_passing_untied_weights(dummy_bmg):
    depth = 3
    mp = MultiweightMessagePassing(depth=depth)
    mp.to("cpu")

    # Get weights from different steps
    weight_step_0 = mp.W_h[0].weight.clone()
    weight_step_1 = mp.W_h[1].weight.clone()

    # They should be different (randomly initialized)
    assert not torch.allclose(weight_step_0, weight_step_1)

def test_multiweight_message_passing_compatibility(dummy_bmg):
    # Ensure it can be used as a replacement for BondMessagePassing
    depth = 3
    mp_multiweight = MultiweightMessagePassing(depth=depth)
    mp_bond = BondMessagePassing(depth=depth)

    mp_multiweight.to("cpu")
    mp_bond.to("cpu")
    mp_bond.to("cpu")

    output_mw = mp_multiweight(dummy_bmg)
    output_bond = mp_bond(dummy_bmg)

    assert output_mw.shape == output_bond.shape
