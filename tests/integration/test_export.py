import pytest
import torch

from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, RegressionFFN, SumAggregation


def make_batch(smiles: list[str]) -> BatchMolGraph:
    dataset = MoleculeDataset([MoleculeDatapoint.from_smi(smi) for smi in smiles])
    return collate_batch(dataset)[0]


@pytest.mark.usefixtures("batch_mol_graph_pytree")
def test_mpnn_export_dynamic_graph_sizes():
    export_graph = make_batch(["C", "CC", "CCC", "CCCC"])
    inference_graph = make_batch(["C", "S", "N", "O"])
    assert export_graph.V.shape[0] != inference_graph.V.shape[0]
    assert inference_graph.E.shape[0] == 0

    message_passing = BondMessagePassing()
    model = MPNN(
        message_passing, SumAggregation(), RegressionFFN(input_dim=message_passing.output_dim)
    ).eval()
    num_atoms = torch.export.Dim("num_atoms")
    num_edges = torch.export.Dim("num_edges")
    dynamic_shapes = {
        "bmg": [{0: num_atoms}, {0: num_edges}, {1: num_edges}, {0: num_edges}, {0: num_atoms}],
        "V_d": None,
        "X_d": None,
    }

    with torch.inference_mode():
        expected = model(inference_graph)

    exported = torch.export.export(
        model,
        (export_graph,),
        kwargs={"V_d": None, "X_d": None},
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    with torch.inference_mode():
        actual = exported.module()(inference_graph, V_d=None, X_d=None)
    torch.testing.assert_close(actual, expected)
