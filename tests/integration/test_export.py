import pytest
import torch

from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.models import MPNN
from chemprop.nn import (
    AtomMessagePassing,
    BondMessagePassing,
    MABAtomMessagePassing,
    MABBondMessagePassing,
    MeanAggregation,
    RegressionFFN,
)
from chemprop.utils.export import add_export_checks, register_batch_mol_graph_pytree

register_batch_mol_graph_pytree()

EXPORT_SMILES = ["C", "CC", "CCC", "CCCC"]
INFERENCE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O"]


def make_batch(smiles: list[str]) -> BatchMolGraph:
    dataset = MoleculeDataset([MoleculeDatapoint.from_smi(smi) for smi in smiles])
    return collate_batch(dataset)[0]


def make_dynamic_shapes() -> dict:
    """Mark the atom and edge dimensions of the five ``BatchMolGraph`` tensors as dynamic."""
    num_atoms = torch.export.Dim("num_atoms")
    num_edges = torch.export.Dim("num_edges")
    return {"bmg": [{0: num_atoms}, {0: num_edges}, {1: num_edges}, {0: num_edges}, {0: num_atoms}]}


def export_and_run_torch(model, export_graph, inference_graph, dynamic_shapes):
    """Export with ``torch.export`` and run the exported program in PyTorch."""
    exported = torch.export.export(
        model, (export_graph,), dynamic_shapes=dynamic_shapes, strict=False
    )
    with torch.inference_mode():
        return exported.module()(inference_graph)


def export_and_run_onnx(model, export_graph, inference_graph, dynamic_shapes):
    """Export to ONNX (``torch.export`` plus conversion to ONNX operators) and run in ONNX Runtime."""
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    onnx_program = torch.onnx.export(
        model, (export_graph,), dynamic_shapes=dynamic_shapes, dynamo=True, verbose=False
    )
    return onnx_program(inference_graph)[0]


@pytest.mark.parametrize(
    "message_passing",
    [AtomMessagePassing(), BondMessagePassing(), MABAtomMessagePassing(), MABBondMessagePassing()],
)
def test_message_passing_export_dynamic_graph_sizes(message_passing):
    export_graph = make_batch(EXPORT_SMILES)
    inference_graph = make_batch(INFERENCE_SMILES)

    exported = torch.export.export(
        message_passing, (export_graph,), dynamic_shapes=make_dynamic_shapes(), strict=False
    )

    expected = message_passing(inference_graph)
    actual = exported.module()(inference_graph)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("inference_smiles", [INFERENCE_SMILES, ["CCO"]])
@pytest.mark.parametrize("export_and_run", [export_and_run_torch, export_and_run_onnx])
def test_exported_mpnn_matches_eager_on_new_batch(export_and_run, inference_smiles):
    export_graph = make_batch(EXPORT_SMILES)
    inference_graph = make_batch(inference_smiles)
    assert inference_graph.V.shape[0] != export_graph.V.shape[0]
    assert len(inference_graph) != len(export_graph)

    model = MPNN(BondMessagePassing(), MeanAggregation(), RegressionFFN(), batch_norm=True).eval()
    add_export_checks(model)

    with torch.inference_mode():
        expected = model(inference_graph)

    actual = export_and_run(model, export_graph, inference_graph, make_dynamic_shapes())
    torch.testing.assert_close(actual, expected)


def test_add_export_checks_hooks_only_batch_norm():
    """Hooks go on batch norm layers only, and the returned handles remove them."""
    model = MPNN(BondMessagePassing(), MeanAggregation(), RegressionFFN(), batch_norm=True)
    handles = add_export_checks(model)
    assert len(handles) == 1
    assert [m for m in model.modules() if m._forward_pre_hooks] == [model.bn]

    for handle in handles:
        handle.remove()
    assert not model.bn._forward_pre_hooks

    assert add_export_checks(MPNN(BondMessagePassing(), MeanAggregation(), RegressionFFN())) == []
