"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader
import torchmetrics

from chemprop import nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch

pytestmark = [
    pytest.mark.parametrize(
        "classification_mpnn_multiclass",
        [nn.BondMessagePassing(), nn.AtomMessagePassing()],
        indirect=True,
    ),
    pytest.mark.integration,
]


@pytest.fixture
def data(mol_classification_data_multiclass):
    smis, Y = mol_classification_data_multiclass

    return [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, Y)]


@pytest.fixture
def dataloader(data):
    dset = MoleculeDataset(data)

    return DataLoader(dset, 32, collate_fn=collate_batch)


def test_quick(classification_mpnn_multiclass, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(classification_mpnn_multiclass, dataloader, None)


def test_overfit(classification_mpnn_multiclass, dataloader):
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
    trainer.fit(classification_mpnn_multiclass, dataloader)

    predss = []
    targetss = []
    for batch in dataloader:
        bmg, _, _, targets, *_ = batch
        preds = classification_mpnn_multiclass(bmg)
        preds = preds.transpose(1, 2)
        predss.append(preds)
        targetss.append(targets)

    preds = torch.cat(predss)
    targets = torch.cat(targetss)
    accuracy = torchmetrics.functional.accuracy(
        preds, targets.long(), task="multiclass", num_classes=3
    )
    assert accuracy > 0.99
