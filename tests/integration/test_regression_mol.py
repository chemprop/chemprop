"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import models, nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch


@pytest.fixture
def data(mol_regression_data):
    smis, Y = mol_regression_data

    return [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, Y)]


@pytest.fixture
def dataloader(data):
    dset = MoleculeDataset(data)
    dset.normalize_targets()

    return DataLoader(dset, 32, collate_fn=collate_batch)


@pytest.mark.parametrize(
    "mpnn",
    [
        (nn.BondMessagePassing(), nn.MeanAggregation()),
        (nn.AtomMessagePassing(), nn.SumAggregation()),
        (nn.BondMessagePassing(), nn.NormAggregation()),
        (nn.BondMessagePassing(), nn.MeanAggregation(), torch.nn.Softplus()),
    ],
    indirect=True,
)
@pytest.mark.integration
def test_quick(mpnn, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(mpnn, dataloader, None)


@pytest.mark.parametrize(
    "mpnn",
    [
        (nn.BondMessagePassing(), nn.MeanAggregation()),
        (nn.AtomMessagePassing(), nn.SumAggregation()),
        (nn.BondMessagePassing(), nn.NormAggregation()),
        (nn.BondMessagePassing(), nn.MeanAggregation(), torch.nn.Softplus()),
    ],
    indirect=True,
)
@pytest.mark.integration
def test_overfit(mpnn, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        max_epochs=50,
        overfit_batches=1.00,
    )
    trainer.fit(mpnn, dataloader)

    errors = []
    for batch in dataloader:
        bmg, _, _, targets, *_ = batch
        preds = mpnn(bmg)
        errors.append(preds - targets)

    errors = torch.cat(errors)
    mse = errors.square().mean().item()

    assert mse <= 0.05


@pytest.mark.parametrize(
    "regression_mpnn_mve", [nn.BondMessagePassing(), nn.AtomMessagePassing()], indirect=True
)
@pytest.mark.integration
def test_mve_quick(regression_mpnn_mve, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(regression_mpnn_mve, dataloader, None)


@pytest.mark.parametrize(
    "regression_mpnn_evidential", [nn.BondMessagePassing(), nn.AtomMessagePassing()], indirect=True
)
@pytest.mark.integration
def test_evidential_quick(regression_mpnn_evidential, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(regression_mpnn_evidential, dataloader, None)


@pytest.mark.parametrize(
    "regression_mpnn_quantile", [nn.BondMessagePassing(), nn.AtomMessagePassing()], indirect=True
)
@pytest.mark.integration
def test_quantile_quick(regression_mpnn_quantile, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(regression_mpnn_quantile, dataloader, None)


def test_input_not_mutated(dataloader):
    V_f_transform = nn.ScaleTransform(mean=[1.0, 2.0], scale=[3.0, 4.0], pad=70)
    E_f_transform = nn.ScaleTransform(mean=[0.0, 1.0], scale=[2.0, 3.0], pad=12)
    graph_transform = nn.GraphTransform(V_transform=V_f_transform, E_transform=E_f_transform)
    V_d_transform = nn.ScaleTransform(mean=[4.0, 5.0], scale=[6.0, 7.0])
    mp = nn.BondMessagePassing(graph_transform=graph_transform, V_d_transform=V_d_transform)

    output_transform = nn.UnscaleTransform(mean=[8.0, 9.0], scale=[10.0, 11.0])
    ffn = nn.RegressionFFN(output_transform=output_transform)

    X_d_transform = nn.ScaleTransform(mean=[13.0, 14.0], scale=[15.0, 16.0])
    model = models.MPNN(mp, nn.NormAggregation(), ffn, X_d_transform=X_d_transform)

    batch = next(iter(dataloader))
    bmg, *_ = batch

    batch2 = next(iter(dataloader))
    bmg2, *_ = batch2

    model.eval()
    _ = model(bmg)

    assert torch.allclose(bmg.V, bmg2.V)
    assert torch.allclose(bmg.E, bmg2.E)
    assert torch.allclose(bmg.edge_index, bmg2.edge_index)
    assert torch.allclose(bmg.rev_edge_index, bmg2.rev_edge_index)
    assert torch.allclose(bmg.batch, bmg2.batch)
