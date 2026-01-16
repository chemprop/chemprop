from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.models import MPNN
from chemprop.uncertainty.estimator import (
    ClassificationDirichletEstimator,
    DropoutEstimator,
    EnsembleEstimator,
    EvidentialAleatoricEstimator,
    EvidentialEpistemicEstimator,
    EvidentialTotalEstimator,
    MulticlassDirichletEstimator,
    MVEEstimator,
    NoUncertaintyEstimator,
    QuantileRegressionEstimator,
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


def test_NoUncertaintyEstimator(data_dir, dataloader, trainer):
    model = MPNN.load_from_file(data_dir / "example_model_v2_regression_mol.pt")
    estimator = NoUncertaintyEstimator()
    preds, uncs = estimator(dataloader, [model], trainer)

    torch.testing.assert_close(preds, torch.tensor([[[2.25354], [2.23501]]]))
    assert uncs is None


def test_DropoutEstimator(data_dir, dataloader, trainer):
    model = MPNN.load_from_file(data_dir / "example_model_v2_regression_mol.pt")
    estimator = DropoutEstimator(ensemble_size=2, dropout=0.1)
    preds, uncs = estimator(dataloader, [model], trainer)

    assert torch.all(uncs != 0)
    assert getattr(model.message_passing.dropout, "p", None) == 0.0


def test_EnsembleEstimator(data_dir, dataloader, trainer):
    model1 = MPNN.load_from_file(data_dir / "example_model_v2_regression_mol.pt")
    model2 = MPNN.load_from_file(data_dir / "example_model_v2_regression_mol.pt")

    # Make the second model predict different values than the first
    model2.predictor.output_transform = torch.nn.Identity()

    estimator = EnsembleEstimator()
    preds, uncs = estimator(dataloader, [model1, model2], trainer)

    torch.testing.assert_close(
        preds, torch.tensor([[[2.25354], [2.23501]], [[0.09652], [0.08291]]])
    )
    torch.testing.assert_close(uncs, torch.tensor([[[1.16318], [1.15788]]]))


def test_EnsembleEstimator_wrong_n_models():
    estimator = EnsembleEstimator()
    with pytest.raises(ValueError):
        estimator("mock_dataloader", ["mock_model"], "mock_trainer")


def test_MVEEstimator(data_dir, dataloader, trainer):
    model = MPNN.load_from_file(data_dir / "example_model_v2_regression_mve_mol.pt")
    estimator = MVEEstimator()
    preds, uncs = estimator(dataloader, [model], trainer)

    torch.testing.assert_close(preds, torch.tensor([[[2.10946], [2.10234]]]))
    torch.testing.assert_close(uncs, torch.tensor([[[1.27602], [1.28058]]]))


@pytest.mark.parametrize(
    "estimator_class, expected_preds, expected_uncs",
    [
        (
            EvidentialTotalEstimator,
            torch.tensor([[[2.09985], [2.09525]]]),
            torch.tensor([[[4.63703], [4.67548]]]),
        ),
        (
            EvidentialEpistemicEstimator,
            torch.tensor([[[2.09985], [2.09525]]]),
            torch.tensor([[[2.77602], [2.80313]]]),
        ),
        (
            EvidentialAleatoricEstimator,
            torch.tensor([[[2.09985], [2.09525]]]),
            torch.tensor([[[1.86101], [1.87234]]]),
        ),
    ],
)
def test_EvidentialEstimators(
    estimator_class, expected_preds, expected_uncs, data_dir, dataloader, trainer
):
    model = MPNN.load_from_file(data_dir / "example_model_v2_regression_evidential_mol.pt")

    estimator = estimator_class()
    preds, uncs = estimator(dataloader, [model], trainer)

    torch.testing.assert_close(preds, expected_preds)
    torch.testing.assert_close(uncs, expected_uncs)


def test_QuantileRegressionEstimator(data_dir, dataloader, trainer):
    model = MPNN.load_from_file(data_dir / "example_model_v2_regression_quantile_mol.pt")
    estimator = QuantileRegressionEstimator()
    preds, uncs = estimator(dataloader, [model], trainer)

    torch.testing.assert_close(preds, torch.tensor([[[2.183332], [2.2001247]]]))
    torch.testing.assert_close(uncs, torch.tensor([[[0.29111385], [0.3591898]]]))


def test_ClassificationDirichletEstimator(data_dir, dataloader, trainer):
    model = MPNN.load_from_file(data_dir / "example_model_v2_classification_dirichlet_mol.pt")
    estimator = ClassificationDirichletEstimator()
    preds, uncs = estimator(dataloader, [model], trainer)

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


def test_MulticlassDirichletEstimator(data_dir, dataloader, trainer):
    model = MPNN.load_from_file(data_dir / "example_model_v2_multiclass_dirichlet_mol.pt")
    estimator = MulticlassDirichletEstimator()
    preds, uncs = estimator(dataloader, [model], trainer)

    torch.testing.assert_close(
        preds, torch.tensor([[[[0.906426, 0.046787, 0.046787]], [[0.925395, 0.037303, 0.037303]]]])
    )
    torch.testing.assert_close(uncs, torch.tensor([[[0.140361], [0.111908]]]))
