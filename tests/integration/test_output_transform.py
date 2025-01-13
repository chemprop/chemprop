from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.models import MPNN


@pytest.fixture
def data(mol_regression_data):
    smis, Y = mol_regression_data

    return [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, Y)]


def test_output_transform(data):
    train_dset = MoleculeDataset(data)
    output_scaler = train_dset.normalize_targets()
    train_loader = DataLoader(train_dset, 32, collate_fn=collate_batch)

    test_dset = MoleculeDataset(data)
    test_loader = DataLoader(test_dset, 32, collate_fn=collate_batch, shuffle=False)

    output_transform = nn.UnscaleTransform.from_standard_scaler(output_scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform)
    mpnn = MPNN(nn.BondMessagePassing(), nn.MeanAggregation(), ffn)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        max_epochs=100,
        overfit_batches=1.00,
    )
    trainer.fit(mpnn, train_loader)

    mpnn.train()
    predss = []
    for batch in train_loader:
        bmg, _, _, targets, *_ = batch
        preds = mpnn(bmg)
        predss.append(preds)

    preds = torch.cat(predss)
    std, mean = torch.std_mean(preds, dim=0)

    assert torch.allclose(std, torch.ones_like(std), atol=0.1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=0.1)

    predss = trainer.predict(mpnn, test_loader)
    preds = torch.cat(predss)
    std, mean = torch.std_mean(preds, dim=0)
    y_std, y_mean = torch.std_mean(torch.from_numpy(test_dset.Y).float(), dim=0)

    assert torch.allclose(std, y_std, atol=0.1)
    assert torch.allclose(mean, y_mean, atol=0.1)
