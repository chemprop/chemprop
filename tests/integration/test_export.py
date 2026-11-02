import numpy as np
import pytest
import torch

from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN, SumAggregation


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


@pytest.mark.usefixtures("batch_mol_graph_pytree")
def test_mean_aggregation_onnx_parity_dynamic_batch(tmp_path):
    ort = pytest.importorskip("onnxruntime")

    export_smiles = ["c1ccccc1", "CC(=O)O"]
    infer_smiles = ["C", "CC", "CCC", "CCO", "CCN", "CCCl", "c1ccccc1", "CC(=O)O", "CCOC", "CCBr"]
    export_graph = make_batch(export_smiles)
    infer_graph = make_batch(infer_smiles)
    assert export_graph.V.shape[0] < infer_graph.V.shape[0]
    assert len(export_smiles) < len(infer_smiles)

    message_passing = BondMessagePassing()
    model = MPNN(
        message_passing, MeanAggregation(), RegressionFFN(input_dim=message_passing.output_dim)
    ).eval()

    auto = torch.export.Dim.AUTO
    dynamic_shapes = {
        "bmg": [{0: auto, 1: auto}, {0: auto, 1: auto}, {0: auto, 1: auto}, {0: auto}, {0: auto}],
        "V_d": None,
        "X_d": None,
    }

    exported = torch.export.export(
        model,
        (export_graph,),
        kwargs={"V_d": None, "X_d": None},
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    onnx_path = tmp_path / "mean_agg.onnx"
    onnx_program = torch.onnx.export(exported, f=str(onnx_path), dynamo=True)

    with torch.inference_mode():
        pt_out = model(infer_graph).detach().cpu().numpy().reshape(-1)

    sess = ort.InferenceSession(
        onnx_program.model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feed = {
        "bmg_v": infer_graph.V.detach().cpu().numpy(),
        "bmg_e": infer_graph.E.detach().cpu().numpy(),
        "bmg_edge_index": infer_graph.edge_index.detach().cpu().numpy(),
        "bmg_rev_edge_index": infer_graph.rev_edge_index.detach().cpu().numpy(),
        "bmg_batch": infer_graph.batch.detach().cpu().numpy(),
    }
    input_names = [inp.name for inp in sess.get_inputs()]
    ort_out = sess.run(None, {name: feed[name] for name in input_names})[0].reshape(-1)

    assert np.max(np.abs(pt_out - ort_out)) <= 1e-4
