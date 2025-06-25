from typing import Sequence, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from chemprop.models.model import MPNN
from chemprop.data.datasets import MoleculeDataset
from chemprop.cli.utils.parsing import make_datapoints, make_dataset
from chemprop.data.collate import collate_batch
from chemprop.sklearn_integration.transformer import build_default_mpnn


class ChempropTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: Sequence[str], y=None):
        return self

    def fit_transform(self, X, y=None, **_):
        return self._build_dataset(X, y)

    def transform(self, X: Sequence[str]):
        return self._build_dataset(X, None)

    def _build_dataset(self, X: Sequence[str], Y: Optional[np.ndarray]) -> MoleculeDataset:
        smiss = [list(X)]
        if Y is None:
            Y = np.zeros((len(X), 1), dtype=float)
        mol_data, _ = make_datapoints(
            smiss=smiss,
            rxnss=None,
            Y=Y,
            weights=None,
            lt_mask=None,
            gt_mask=None,
            X_d=None,
            V_fss=None,
            E_fss=None,
            V_dss=None,
            molecule_featurizers=None,
            keep_h=False,
            add_h=False,
            ignore_stereo=False,
            reorder_atoms=True,
        )
        return make_dataset(mol_data[0])


class ChempropRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        mpnn_args: Optional[dict] = None,
        n_epochs: int = 30,
        batch_size: int = 32,
        num_workers: int = 0,
        device: str = "cpu",
    ):
        self.mpnn_args = mpnn_args
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.model = None
        self.trainer = Trainer(
            max_epochs=self.n_epochs,
            accelerator="gpu" if torch.cuda.is_available() and "cuda" in self.device else "cpu",
            devices=1,
            logger=False,
            callbacks=[
                ModelCheckpoint(monitor="train_loss", mode="min"),
                EarlyStopping(monitor="train_loss", patience=5, mode="min"),
            ],
        )

    def fit(self, X: MoleculeDataset, y=None):
        g0 = X[0].mg
        self.model = (
            MPNN(**self.mpnn_args)
            if self.mpnn_args
            else build_default_mpnn(g0.V.shape[1], g0.E.shape[1])
        )
        dl = DataLoader(
            X,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
        )
        self.trainer.fit(self.model, dl)
        return self

    def predict(self, X: MoleculeDataset):
        self.model.eval()
        dl = DataLoader(
            X,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
        )
        preds = self.trainer.predict(self.model, dataloaders=dl)
        return torch.cat(preds,dim=0).view(-1).cpu().numpy()


if __name__ == "__main__":
    # microtest
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    sklearnPipeline = Pipeline(
        [("featurizer", ChempropTransformer()), ("regressor", ChempropRegressor())]
    )

    X = np.array(
        [
            "CCO",
            "CCN",
            "CCC",
            "COC",
            "CNC",
            "CCCl",
            "CCBr",
            "CCF",
            "CCI",
            "CC=O",
            "CC#N",
            "CC(C)O",
            "CC(C)N",
            "CC(C)C",
            "COC(C)",
            "CN(C)C",
            "C1CCCCC1",
            "C1=CC=CC=C1",
            "CC(C)(C)O",
            "CC(C)(C)N",
            "COCCO",
            "CCOC(=O)C",
            "CCN(CC)CC",
            "CN1CCCC1",
            "C(CO)N",
        ]
    )

    y = np.array(
        [
            0.50,
            0.60,
            0.55,
            0.58,
            0.52,
            0.62,
            0.65,
            0.57,
            0.59,
            0.61,
            0.56,
            0.60,
            0.54,
            0.53,
            0.62,
            0.63,
            0.45,
            0.40,
            0.64,
            0.66,
            0.59,
            0.51,
            0.48,
            0.46,
            0.49,
        ]
    )

    scores = cross_val_score(sklearnPipeline, X, y, cv=5, scoring="neg_mean_squared_error")
    print("Cross-validation scores:", scores)
    print("Mean MSE:", -scores.mean())

    # microtest
    sklearnPipeline1 = Pipeline(
        [("featurizer", ChempropTransformer()), ("regressor", ChempropRegressor())]
    )

    X_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
    y_targets = [0.5, 1.2, 0.7]
    sklearnPipeline1.fit(X_smiles, y_targets)

    X_test = ["CCN", "CCCl"]
    predictions = sklearnPipeline1.predict(X_test)
    print("Predictions:", predictions)

    import joblib

    joblib.dump(sklearnPipeline1, "chemprop_pipeline.pkl")

    loadedPipeline = joblib.load("chemprop_pipeline.pkl")
    print("Reproduced Predictions:", loadedPipeline.predict(X))
