from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.models import MPNN
from chemprop.uncertainty.predictor import (
    DropoutPredictor,
    EnsemblePredictor,
    EvidentialAleatoricPredictor,
    EvidentialEpistemicPredictor,
    EvidentialTotalPredictor,
    MVEPredictor,
    NoUncertaintyPredictor,
)


@pytest.fixture
def dataloader(mol_regression_data):
    smis, Y = mol_regression_data
    data = [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis[:2], Y[:2])]
    dset = MoleculeDataset(data)

    return DataLoader(dset, 32, collate_fn=collate_batch)


@pytest.fixture
def trainer():
    return pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )


def test_NoUncertaintyPredictor(data_dir, dataloader, trainer):
    model = MPNN.load_from_file(data_dir / "example_model_v2_regression_mol.pt")
    predictor = NoUncertaintyPredictor()
    preds, uncs = predictor(dataloader, [model], trainer)

    torch.testing.assert_close(preds, torch.tensor([[[2.25354], [2.23501]]]))
    assert uncs is None


def test_DropoutPredictor(data_dir, dataloader, trainer):
    model = MPNN.load_from_file(data_dir / "example_model_v2_regression_mol.pt")
    predictor = DropoutPredictor(ensemble_size=2, dropout=0.1)
    preds, uncs = predictor(dataloader, [model], trainer)

    assert torch.all(uncs != 0)
    assert getattr(model.message_passing.dropout, "p", None) == 0.0


def test_EnsemblePredictor(data_dir, dataloader, trainer):
    model1 = MPNN.load_from_file(data_dir / "example_model_v2_regression_mol.pt")
    model2 = MPNN.load_from_file(data_dir / "example_model_v2_regression_mol.pt")

    model2.predictor.output_transform = torch.nn.Identity()

    predictor = EnsemblePredictor()
    preds, uncs = predictor(dataloader, [model1, model2], trainer)

    torch.testing.assert_close(
        preds, torch.tensor([[[2.25354], [2.23501]], [[0.09652], [0.08291]]])
    )
    torch.testing.assert_close(uncs, torch.tensor([[[1.16318], [1.15788]]]))


def test_EnsemblePredictor_wrong_n_models():
    predictor = EnsemblePredictor()
    with pytest.raises(ValueError):
        predictor("mock_dataloader", ["mock_model"], "mock_trainer")


def test_MVEPredictor(data_dir, dataloader, trainer):
    model = MPNN.load_from_file(data_dir / "example_model_v2_regression_mve_mol.pt")
    predictor = MVEPredictor()
    preds, uncs = predictor(dataloader, [model], trainer)

    torch.testing.assert_close(preds, torch.tensor([[[2.10946], [2.10234]]]))
    torch.testing.assert_close(uncs, torch.tensor([[[1.27602], [1.28058]]]))


@pytest.mark.parametrize(
    "predictor_class, expected_preds, expected_uncs",
    [
        (
            EvidentialTotalPredictor,
            torch.tensor([[[2.09985], [2.09525]]]),
            torch.tensor([[[4.63703], [4.67548]]]),
        ),
        (
            EvidentialEpistemicPredictor,
            torch.tensor([[[2.09985], [2.09525]]]),
            torch.tensor([[[2.77602], [2.80313]]]),
        ),
        (
            EvidentialAleatoricPredictor,
            torch.tensor([[[2.09985], [2.09525]]]),
            torch.tensor([[[1.86101], [1.87234]]]),
        ),
    ],
)
def test_EvidentialPredictors(
    predictor_class, expected_preds, expected_uncs, data_dir, dataloader, trainer
):
    model = MPNN.load_from_file(data_dir / "example_model_v2_regression_evidential_mol.pt")

    predictor = predictor_class()
    preds, uncs = predictor(dataloader, [model], trainer)

    torch.testing.assert_close(preds, expected_preds)
    torch.testing.assert_close(uncs, expected_uncs)
