"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

from lightning import pytorch as pl
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN


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


def test_input_not_mutated():
    smis = ["CC", "CN", "CO", "CF", "CP", "CS", "CI"]
    ys = np.random.rand(len(smis), 1) * 100

    n_datapoints = len(smis)
    n_atoms = 2
    n_bonds = 1
    n_extra_atom_features = 3
    n_extra_bond_features = 4
    n_extra_atom_descriptors = 5
    n_extra_datapoint_descriptors = 6

    extra_atom_features = np.random.rand(n_datapoints, n_atoms, n_extra_atom_features)
    extra_bond_features = np.random.rand(n_datapoints, n_bonds, n_extra_bond_features)
    extra_atom_descriptors = np.random.rand(n_datapoints, n_atoms, n_extra_atom_descriptors)
    extra_datapoint_descriptors = np.random.rand(n_datapoints, n_extra_datapoint_descriptors)

    datapoints = [
        MoleculeDatapoint.from_smi(smi, y, x_d=x_d, V_f=V_f, E_f=E_f, V_d=V_d)
        for smi, y, x_d, V_f, E_f, V_d in zip(
            smis,
            ys,
            extra_datapoint_descriptors,
            extra_atom_features,
            extra_bond_features,
            extra_atom_descriptors,
        )
    ]
    featurizer = SimpleMoleculeMolGraphFeaturizer(
        extra_atom_fdim=n_extra_atom_features, extra_bond_fdim=n_extra_bond_features
    )
    dset = MoleculeDataset(datapoints, featurizer=featurizer)
    dataloader = DataLoader(dset, 32, collate_fn=collate_batch)

    V_f_transform = nn.ScaleTransform(
        mean=np.random.rand(n_extra_atom_features),
        scale=np.random.rand(n_extra_atom_features),
        pad=72,
    )
    E_f_transform = nn.ScaleTransform(
        mean=np.random.rand(n_extra_bond_features),
        scale=np.random.rand(n_extra_bond_features),
        pad=14,
    )
    graph_transform = nn.GraphTransform(V_transform=V_f_transform, E_transform=E_f_transform)
    V_d_transform = nn.ScaleTransform(
        mean=np.random.rand(n_extra_atom_descriptors),
        scale=np.random.rand(n_extra_atom_descriptors),
    )
    mp = nn.BondMessagePassing(
        graph_transform=graph_transform,
        V_d_transform=V_d_transform,
        d_v=featurizer.atom_fdim,
        d_e=featurizer.bond_fdim,
        d_vd=n_extra_atom_descriptors,
    )

    output_transform = nn.UnscaleTransform(mean=np.random.rand(1), scale=np.random.rand(1))
    ffn = nn.RegressionFFN(
        output_transform=output_transform, input_dim=(mp.output_dim + n_extra_datapoint_descriptors)
    )

    X_d_transform = nn.ScaleTransform(
        mean=np.random.rand(n_extra_datapoint_descriptors),
        scale=np.random.rand(n_extra_datapoint_descriptors),
    )
    model = MPNN(mp, nn.NormAggregation(), ffn, X_d_transform=X_d_transform)

    batch = next(iter(dataloader))
    bmg, V_d, X_d, *_ = batch

    batch2 = next(iter(dataloader))
    bmg2, V_d2, X_d2, *_ = batch2

    model.eval()
    _ = model(bmg, V_d, X_d)

    assert torch.allclose(bmg.V, bmg2.V)
    assert torch.allclose(bmg.E, bmg2.E)
    assert torch.allclose(bmg.edge_index, bmg2.edge_index)
    assert torch.allclose(bmg.rev_edge_index, bmg2.rev_edge_index)
    assert torch.allclose(bmg.batch, bmg2.batch)
    assert torch.allclose(V_d, V_d2)
    assert torch.allclose(X_d, X_d2)
