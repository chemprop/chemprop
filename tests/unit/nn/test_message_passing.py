import numpy as np
import pytest
import torch

from chemprop.data import BatchMolGraph
from chemprop.data.molgraph import MolGraph
from chemprop.nn import (
    AtomMessagePassing,
    BondMessagePassing,
    MABAtomMessagePassing,
    MABBondMessagePassing,
)


def make_chain_graph(num_atoms: int) -> BatchMolGraph:
    src = np.arange(num_atoms - 1)
    dst = src + 1
    edge_index = np.stack((np.concatenate((src, dst)), np.concatenate((dst, src))))
    num_edges = edge_index.shape[1]
    mol_graph = MolGraph(
        V=np.ones((num_atoms, 72), dtype=np.float32),
        E=np.ones((num_edges, 14), dtype=np.float32),
        edge_index=edge_index,
        rev_edge_index=np.concatenate((np.arange(num_atoms - 1, num_edges), src)),
    )
    return BatchMolGraph([mol_graph])


@pytest.mark.parametrize(
    "message_passing",
    [AtomMessagePassing(), BondMessagePassing(), MABAtomMessagePassing(), MABBondMessagePassing()],
)
def test_message_passing_export_dynamic_graph_sizes(message_passing, batch_mol_graph_pytree):
    export_graph = make_chain_graph(3)
    inference_graph = make_chain_graph(5)
    num_atoms = torch.export.Dim("num_atoms", min=2)
    num_edges = torch.export.Dim("num_edges", min=2)
    dynamic_shapes = {
        "bmg": [{0: num_atoms}, {0: num_edges}, {1: num_edges}, {0: num_edges}, {0: num_atoms}]
    }

    exported = torch.export.export(
        message_passing, (export_graph,), dynamic_shapes=dynamic_shapes, strict=False
    )

    expected = message_passing(inference_graph)
    actual = exported.module()(inference_graph)
    torch.testing.assert_close(actual, expected)
