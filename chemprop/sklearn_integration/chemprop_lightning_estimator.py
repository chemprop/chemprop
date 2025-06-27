from typing import Sequence, Optional, Literal
import numpy as np
import torch
from torch.utils.data import DataLoader
from os import PathLike
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentError, ArgumentParser, Namespace

from chemprop.models.model import MPNN
from chemprop.data.datasets import MoleculeDataset, ReactionDataset
from chemprop.cli.utils.parsing import make_datapoints, make_dataset
from chemprop.data.collate import collate_batch
from chemprop.sklearn_integration.transformer import build_default_mpnn
from chemprop.featurizers.molgraph.reaction import RxnMode
from chemprop.cli.train import build_model, build_MAB_model
from chemprop.models.utils import save_model
from chemprop.cli.common import add_common_args
from chemprop.cli.train import add_train_args


class ChempropMoleculeTransformer(BaseEstimator, TransformerMixin):
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
        elif Y.ndim == 1:
            Y = Y.reshape(-1,1)
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


class ChempropReactionTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        reaction_mode = RxnMode.REAC_DIFF,
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = True,
    ):
        self.reaction_mode = reaction_mode
        self.keep_h = keep_h
        self.add_h = add_h
        self.ignore_stereo = ignore_stereo
        self.reorder_atoms = reorder_atoms

    def fit(self, X: Sequence[str], y=None):
        return self

    def fit_transform(self, X: Sequence[str], y=None, **__):
        return self._build_dataset(X, y)

    def transform(self, X: Sequence[str]):
        return self._build_dataset(X, None)

    def _build_dataset(self, X: Sequence[str], Y: Optional[Sequence[float]]) -> ReactionDataset:
        rxnss = [list(X)]
        if Y is None:
            Y = np.zeros((len(X), 1), dtype=float)
        elif Y.ndim == 1:
            Y = Y.reshape(-1,1)
        _, rxn_data = make_datapoints(
            smiss=None,
            rxnss=rxnss,
            Y=Y,
            weights=None,
            lt_mask=None,
            gt_mask=None,
            X_d=None,
            V_fss=None,
            E_fss=None,
            V_dss=None,
            molecule_featurizers=None,
            keep_h=self.keep_h,
            add_h=self.add_h,
            ignore_stereo=self.ignore_stereo,
            reorder_atoms=self.reorder_atoms,
        )

        return make_dataset(
            rxn_data[0],
            reaction_mode=self.reaction_mode,
        )

def add_train_defaults(args: Namespace) -> Namespace:
    parser = ArgumentParser()
    parser = add_common_args(parser)
    parser = add_train_args(parser)
    defaults = parser.parse_args([])
    for k, v in vars(defaults).items():
        if not hasattr(args,k):
            setattr(args,k,v)
    return args

class ChempropRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        args: Namespace = Namespace(),
        n_epochs: int = 50,
        batch_size: int = 32,
        num_workers: int = 0,
        device: str = "cpu",
    ):
        self.args = add_train_defaults(args)
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

    def __sklearn_is_fitted__(self):
        return True
    
    def fit(self, X: MoleculeDataset|ReactionDataset, y=None):
        if self.model is None:
            input_transforms = (None, [None], [None], [None])
            output_transform = None
            self.model = build_model(self.args, X, output_transform, input_transforms)
        dl = DataLoader(
            X,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            drop_last=True,
        )
        self.trainer.fit(self.model, dl)
        return self

    def predict(self, X: MoleculeDataset|ReactionDataset):
        dl = DataLoader(
            X,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            drop_last=True,
        )
        preds = self.trainer.predict(self.model, dataloaders=dl)
        return torch.cat(preds,dim=0).view(-1).cpu().numpy()
    
    def load_from_file(self, checkpoint_path: str, strict: bool = True):
        self.model = MPNN.load_from_file(checkpoint_path, map_location=self.device, strict=strict)
        return self
    
    def save_model(self, path: PathLike):
        output_columns = self.mpnn_args.target_columns
        save_model(path, self.model, output_columns)
        return self
        

if __name__ == "__main__":
    # microtest
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score,train_test_split
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    sklearnPipeline = Pipeline(
        [("featurizer", ChempropMoleculeTransformer()), ("regressor", ChempropRegressor())]
    )

    df = pd.read_csv("mol.csv") # change to target datapath
    X = df["smiles"].to_numpy(dtype=str)
    y = df["lipo"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    sklearnPipeline.fit(X_train, y_train)
    y_pred = sklearnPipeline.predict(X_test)
    print(f"MSE: {mean_squared_error(y_true=y_test, y_pred=y_pred)}")

    # scores = cross_val_score(sklearnPipeline, X, y, cv=5, scoring="neg_mean_squared_error")
    # print("Cross-validation scores:", scores)
    # print("Mean MSE:", -scores.mean())


    # # save/reload test
    # sklearnPipeline1 = Pipeline(
    #     [("featurizer", ChempropMoleculeTransformer()), ("regressor", ChempropRegressor())]
    # )

    # X_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
    # y_targets = [0.5, 1.2, 0.7]
    # sklearnPipeline1.fit(X_smiles, y_targets)

    # X_test = ["CCN", "CCCl"]
    # predictions = sklearnPipeline1.predict(X_test)
    # print("Predictions:", predictions)

    # import joblib

    # joblib.dump(sklearnPipeline1, "chemprop_pipeline.pkl")

    # loadedPipeline = joblib.load("chemprop_pipeline.pkl")
    # print("Reproduced Predictions:", loadedPipeline.predict(X))
