"""This integration test is designed to ensure that the chemprop model for atom properties can
_overfit_ the training data. A small enough dataset should be memorizable by even a moderately sized
model, so this test should generally pass."""

import ast

from lightning import pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import nn
from chemprop.data import AtomDataset, MoleculeDatapoint, collate_batch

pytestmark = [
    pytest.mark.parametrize(
        "mpnn",
        [
            (nn.BondMessagePassing(), nn.NoAggregation()),
            (nn.AtomMessagePassing(), nn.NoAggregation()),
        ],
        indirect=True,
    ),
    pytest.mark.integration,
]


@pytest.fixture
def atom_regression_data(data_dir):
    df = pd.read_csv(data_dir / "regression/atoms.csv")
    smis = df.loc[:, "smiles"].values
    target_columns = ["charges"]
    ys = df.loc[:, target_columns]
    Y = []
    for molecule in range(len(ys)):
        list_props = []
        for prop in target_columns:
            np_prop = np.array(ast.literal_eval(ys.iloc[molecule][prop]))
            np_prop = np.expand_dims(np_prop, axis=1)
            list_props.append(np_prop)
        Y.append(np.hstack(list_props))

    return smis, Y


@pytest.fixture
def data(atom_regression_data):
    smis, Y = atom_regression_data

    return [
        MoleculeDatapoint.from_smi(smi, y, keep_h=True, keep_atom_map=True)
        for smi, y in zip(smis, Y)
    ]


@pytest.fixture
def dataloader(data):
    dset = AtomDataset(data)
    dset.normalize_targets()

    return DataLoader(dset, 32, collate_fn=collate_batch)


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


def test_overfit(mpnn, dataloader):
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
    trainer.fit(mpnn, dataloader)

    errors = []
    for batch in dataloader:
        bmg, _, _, targets, *_ = batch
        preds = mpnn(bmg)
        errors.append(preds - targets)

    errors = torch.cat(errors)
    mse = errors.square().mean().item()

    assert mse <= 0.05
