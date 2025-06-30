from typing import Sequence, Optional, Literal, List
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from os import PathLike
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser, Namespace

from chemprop.models.model import MPNN
from chemprop.data.datasets import (
    MoleculeDataset,
    ReactionDataset,
    MolAtomBondDataset,
    MulticomponentDataset,
)
from chemprop.cli.utils.parsing import make_datapoints, make_dataset, parse_csv
from chemprop.data.collate import collate_batch, collate_mol_atom_bond_batch, collate_multicomponent
from chemprop.featurizers.molgraph.reaction import RxnMode
from chemprop.cli.train import build_model
from chemprop.models.utils import save_model
from chemprop.cli.common import add_common_args
from chemprop.cli.train import add_train_args


class ChempropMoleculeTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = False,
        smiles_cols: Sequence[str] | None = None,
        rxn_cols: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
        ignore_cols: Sequence[str] | None = None,
        weight_col: str | None = None,
        no_header_row: bool = False,
    ):
        self.keep_h = keep_h
        self.add_h = add_h
        self.ignore_stereo = ignore_stereo
        self.reorder_atoms = reorder_atoms

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
            Y = Y.reshape(-1, 1)
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
            keep_h=self.keep_h,
            add_h=self.add_h,
            ignore_stereo=self.ignore_stereo,
            reorder_atoms=self.reorder_atoms,
        )
        return make_dataset(mol_data[0])


class ChempropReactionTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        reaction_mode=RxnMode.REAC_DIFF,
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = False,
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
            Y = Y.reshape(-1, 1)
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

        return make_dataset(rxn_data[0], reaction_mode=self.reaction_mode)


class ChempropMulticomponentTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        component_types: Sequence[Literal["molecule", "reaction"]],
        reaction_modes: Sequence[RxnMode] | None = None,
    ):
        self.component_types = list(component_types)
        self.reaction_modes = list(reaction_modes or [])
        self.col_tf = []
        rxn_iter = iter(self.reaction_modes + [RxnMode.REAC_DIFF] * 100)
        for t in self.component_types:
            self.col_tf.append(
                ChempropMoleculeTransformer()
                if t == "molecule"
                else ChempropReactionTransformer(reaction_mode=next(rxn_iter))
            )

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **__):
        return self._build_dataset(X, y)

    def transform(self, X):
        return self._build_dataset(X, None)

    def _build_dataset(self, X_cols, Y):
        X_cols = list(X_cols)
        if Y is None:
            Y = np.zeros((len(X_cols[0]), 1), dtype=float)
        elif Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        datasets = [tf.fit_transform(col, Y) for tf, col in zip(self.col_tf, X_cols)]
        return MulticomponentDataset(datasets)


def add_train_defaults(args: Namespace) -> Namespace:
    parser = ArgumentParser()
    parser = add_common_args(parser)
    parser = add_train_args(parser)
    defaults = parser.parse_args([])
    for k, v in vars(defaults).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def pick_collate(dset):
    if isinstance(dset, MulticomponentDataset):
        return collate_multicomponent
    if isinstance(dset, MolAtomBondDataset):
        return collate_mol_atom_bond_batch
    return collate_batch


class ChempropRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        smiles_columns: Optional[Sequence[str]] = None,
        reaction_columns: Optional[Sequence[str]] = None,
        num_workers: int = 0,
        batch_size: int = 64,
        data_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        checkpoint: Optional[str | Path] = None,
        rxn_mode: str = "REAC_DIFF",
        multi_hot_atom_featurizer_mode: str = "V2",
        molecule_featurizers: Optional[List[str]] = None,
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = False,
        no_descriptor_scaling: bool = False,
        message_hidden_dim: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        aggregation: str = "norm",
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
        batch_norm: bool = False,
        multiclass_num_classes: int = 3,
        accelerator: str = "auto",
        devices: str | int | Sequence[int] = "auto",
        epochs: int = 50,
        patience: Optional[int] = None,
        tracking_metric: str = "val_loss",
        task_type: str = "regression",
        loss_function: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        weight_column: Optional[str] = None,
        target_columns: Optional[Sequence[str]] = None,
        ignore_columns: Optional[Sequence[str]] = None,
    ):
        args = Namespace(
            smiles_columns=smiles_columns,
            reaction_columns=reaction_columns,
            num_workers=num_workers,
            batch_size=batch_size,
            data_path=data_path,
            output_dir=str(output_dir) if output_dir is not None else None,
            checkpoint=str(checkpoint) if checkpoint is not None else None,
            rxn_mode=rxn_mode,
            multi_hot_atom_featurizer_mode=multi_hot_atom_featurizer_mode,
            molecule_featurizers=molecule_featurizers,
            keep_h=keep_h,
            add_h=add_h,
            ignore_stereo=ignore_stereo,
            reorder_atoms=reorder_atoms,
            no_descriptor_scaling=no_descriptor_scaling,
            message_hidden_dim=message_hidden_dim,
            depth=depth,
            dropout=dropout,
            aggregation=aggregation,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_num_layers=ffn_num_layers,
            batch_norm=batch_norm,
            multiclass_num_classes=multiclass_num_classes,
            accelerator=accelerator,
            devices=devices,
            epochs=epochs,
            patience=patience,
            tracking_metric=tracking_metric,
            task_type=task_type,
            loss_function=loss_function,
            metrics=metrics,
            weight_column=weight_column,
            target_columns=target_columns,
            ignore_columns=ignore_columns,
        )
        self.args = add_train_defaults(args)
        self.model = None
        self.trainer = Trainer(
            max_epochs=self.args.epochs,
            callbacks=[
                ModelCheckpoint(monitor="train_loss", mode="min"),
                EarlyStopping(monitor="train_loss", patience=5, mode="min"),
            ],
        )

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X: MoleculeDataset | ReactionDataset | MulticomponentDataset, y=None):
        if self.model is None:
            n_comp = X.n_components if isinstance(X, MulticomponentDataset) else 1
            input_transforms = (None, [None] * n_comp, [None] * n_comp, [None] * n_comp)
            output_transform = None
            self.model = build_model(self.args, X, output_transform, input_transforms)
        dl = DataLoader(
            X,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=pick_collate(X),
            drop_last=True,
        )
        self.trainer.fit(self.model, dl)
        return self

    def predict(self, X: MoleculeDataset | ReactionDataset | MulticomponentDataset):
        dl = DataLoader(
            X,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=pick_collate(X),
            drop_last=True,
        )
        preds = self.trainer.predict(self.model, dataloaders=dl)
        return torch.cat(preds, dim=0).view(-1).cpu().numpy()

    def _load_from_file(self, checkpoint_path: str, strict: bool = True):
        self.model = MPNN.load_from_file(checkpoint_path, strict=strict)
        return self

    def _save_model(self, path: PathLike):
        output_columns = self.args.target_columns
        save_model(path, self.model, output_columns)
        return self



if __name__ == "__main__":
    # microtest
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, train_test_split
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    sklearnPipeline = Pipeline(
        [("featurizer", ChempropMulticomponentTransformer(component_types=["reaction","molecule"])), 
         ("regressor", ChempropRegressor())]
    )

    # df = pd.read_csv("rxn.csv")  # change to target datapath
    # X = df["smiles"].to_numpy(dtype=str)
    # y = df["ea"].to_numpy(dtype=float)

    df = pd.read_csv("rxn+mol.csv")  # change to target datapath
    X = (df["rxn_smiles"].to_numpy(dtype=str), df["solvent_smiles"].to_numpy(dtype=str))
    y = df["target"].to_numpy(dtype=float)
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    sklearnPipeline.fit(X, y)
    y_pred = sklearnPipeline.predict(X)
    print(f"MSE: {mean_squared_error(y_true=y, y_pred=y_pred)}")

    # scores = cross_val_score(sklearnPipeline, X, y, cv=5, scoring="neg_mean_squared_error")
    # print("Cross-validation scores:", scores)
    # print("Mean MSE:", -scores.mean())

    


