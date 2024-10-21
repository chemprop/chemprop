from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.models import MPNN
from chemprop.uncertainty.predictor import (
    ClassificationDirichletPredictor,
    DropoutPredictor,
    EnsemblePredictor,
    MulticlassDirichletPredictor,
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

    # Make the second model predict different values than the first
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


def test_ClassificationDirichletPredictor(data_dir, dataloader, trainer):
    model = MPNN.load_from_file(data_dir / "example_model_v2_classification_dirichlet_mol.pt")
    predictor = ClassificationDirichletPredictor()
    preds, uncs = predictor(dataloader, [model], trainer)

    torch.testing.assert_close(
        preds,
        torch.tensor(
            [[[0.085077, 0.085050, 0.086104, 0.138729], [0.069522, 0.069501, 0.070306, 0.116051]]]
        ),
    )
    torch.testing.assert_close(
        uncs,
        torch.tensor(
            [[[0.170140, 0.170079, 0.172037, 0.277232], [0.139044, 0.138999, 0.140591, 0.232073]]]
        ),
    )


def test_MulticlassDirichletPredictor(data_dir, dataloader, trainer):
    model = MPNN.load_from_file(data_dir / "example_model_v2_multiclass_dirichlet_mol.pt")
    predictor = MulticlassDirichletPredictor()
    preds, uncs = predictor(dataloader, [model], trainer)

    torch.testing.assert_close(
        preds, torch.tensor([[[[0.906426, 0.046787, 0.046787]], [[0.925395, 0.037303, 0.037303]]]])
    )
    torch.testing.assert_close(
        uncs, torch.tensor([[[[0.140361, 0.140361, 0.140361]], [[0.111908, 0.111908, 0.111908]]]])
    )
