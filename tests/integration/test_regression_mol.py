"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

from pathlib import Path
import warnings

from lightning import pytorch as pl
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop.v2 import featurizers, models, nn
from chemprop.v2.data import MoleculeDatapoint, MoleculeDataset, collate_batch

# warnings.simplefilter("ignore", category=UserWarning, append=True)
warnings.filterwarnings("ignore", module=r"lightning.*", append=True)

@pytest.fixture(
    params=[
        (Path("tests/data/regression_mol.csv"), "smiles", "lipo"),
    ]
)
def data(request):
    p_data, key_col, val_col = request.param
    df = pd.read_csv(p_data)
    smis = df[key_col].to_list()
    Y = df[val_col].to_numpy().reshape(-1, 1)

    return [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, Y)]


@pytest.fixture(params=[nn.BondMessagePassing(), nn.AtomMessagePassing()])
def mp(request):
    return request.param


@pytest.fixture
def dataloader(data):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = MoleculeDataset(data, featurizer)
    dset.normalize_targets()

    return DataLoader(dset, 20, collate_fn=collate_batch)


def test_quick(mp, dataloader):
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN()
    mpnn = models.MPNN(mp, agg, ffn, True)

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


def test_overfit(mp, dataloader):
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN()
    mpnn = models.MPNN(mp, agg, ffn, True)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        max_epochs=100,
        overfit_batches=1.00
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