from typing import NamedTuple

import pytest
import torch
from torch import Tensor

from chemprop.nn import (
    AtomMessagePassing,
    BondMessagePassing,
    MABAtomMessagePassing,
    MABBondMessagePassing,
)


class MockBatchMolGraph(NamedTuple):
    V: Tensor
    E: Tensor
    edge_index: Tensor
    rev_edge_index: Tensor
    batch: Tensor


def make_chain_graph(num_atoms: int) -> MockBatchMolGraph:
    src = torch.arange(num_atoms - 1)
    dst = src + 1
    edge_index = torch.stack((torch.cat((src, dst)), torch.cat((dst, src))))
    num_edges = edge_index.shape[1]

    return MockBatchMolGraph(
        V=torch.randn(num_atoms, 72),
        E=torch.randn(num_edges, 14),
        edge_index=edge_index,
        rev_edge_index=torch.cat((torch.arange(num_atoms - 1, num_edges), src)),
        batch=torch.zeros(num_atoms, dtype=torch.long),
    )


@pytest.mark.parametrize(
    "message_passing",
    [AtomMessagePassing(), BondMessagePassing(), MABAtomMessagePassing(), MABBondMessagePassing()],
)
def test_message_passing_export_dynamic_graph_sizes(message_passing):
    export_graph = make_chain_graph(3)
    inference_graph = make_chain_graph(5)
    num_atoms = torch.export.Dim("num_atoms", min=2)
    num_edges = torch.export.Dim("num_edges", min=2)
    dynamic_shapes = {
        "bmg": MockBatchMolGraph(
            {0: num_atoms}, {0: num_edges}, {1: num_edges}, {0: num_edges}, {0: num_atoms}
        )
    }

    exported = torch.export.export(
        message_passing, (export_graph,), dynamic_shapes=dynamic_shapes, strict=False
    )

    expected = message_passing(inference_graph)
    actual = exported.module()(inference_graph)
    torch.testing.assert_close(actual, expected)
