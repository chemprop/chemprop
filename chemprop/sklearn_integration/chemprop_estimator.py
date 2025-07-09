from typing import Sequence, Optional, Literal, List
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
from os import PathLike
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score, r2_score
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser, Namespace

from chemprop.models import MulticomponentMPNN, MPNN
from chemprop.data.datasets import (
    MoleculeDataset,
    ReactionDataset,
    MolAtomBondDataset,
    MulticomponentDataset,
)
from chemprop.cli.utils.parsing import make_datapoints, make_dataset, parse_csv
from chemprop.data.collate import collate_batch, collate_mol_atom_bond_batch, collate_multicomponent
from chemprop.featurizers.molgraph.reaction import RxnMode
from chemprop.cli.train import build_model, normalize_inputs
from chemprop.models.utils import save_model
from chemprop.cli.common import add_common_args, find_models
from chemprop.cli.train import add_train_args
from chemprop.nn.transforms import UnscaleTransform


logger = logging.getLogger(__name__)


class ChempropMoleculeTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        data_path: Optional[PathLike] = None,
        multi_hot_atom_featurizer_mode: Literal["V1", "V2", "ORGANIC", "RIGR"] = "V2",
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = False,
        smiles_cols: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
        ignore_cols: Sequence[str] | None = None,
        weight_col: str | None = None,
        bounded=None,
        no_header_row: bool = False,
    ):
        self.data_path = data_path
        self.multi_hot_atom_featurizer_mode = multi_hot_atom_featurizer_mode
        self.keep_h = keep_h
        self.add_h = add_h
        self.ignore_stereo = ignore_stereo
        self.reorder_atoms = reorder_atoms
        self.smiles_cols = smiles_cols
        self.target_cols = target_cols
        self.ignore_cols = ignore_cols
        self.weight_col = weight_col
        self.bounded = bounded
        self.no_header_row = no_header_row

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X: Optional[Sequence[str]] = None, y=None, **_):
        return self._build_dataset(X, y)

    def transform(self, X: Optional[Sequence[str]] = None):
        return self._build_dataset(X, None)

    def _build_dataset(
        self,
        X: Optional[Sequence[str]],
        Y: Optional[np.ndarray],
        weights=None,
        lt_mask=None,
        gt_mask=None,
    ) -> MoleculeDataset:
        if self.data_path is not None:
            smiss, _, Y, weights, lt_mask, gt_mask = parse_csv(
                path=self.data_path,
                smiles_cols=self.smiles_cols,
                rxn_cols=None,
                target_cols=self.target_cols,
                ignore_cols=self.ignore_cols,
                splits_col=None,
                weight_col=self.weight_col,
                bounded=self.bounded,
                no_header_row=self.no_header_row,
            )

        else:
            if X is None:
                raise ValueError("No data supplied, X and data_path cannot both be None.")
            smiss = [list(X)]
            if Y is None:
                Y = np.zeros((len(X), 1), dtype=float)
            elif Y.ndim == 1:
                Y = Y.reshape(-1, 1)

        mol_data, _ = make_datapoints(
            smiss=smiss,
            rxnss=None,
            Y=Y,
            weights=weights,
            lt_mask=lt_mask,
            gt_mask=gt_mask,
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
            mol_data[0], multi_hot_atom_featurizer_mode=self.multi_hot_atom_featurizer_mode
        )


class ChempropReactionTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        data_path: Optional[PathLike] = None,
        reaction_mode=RxnMode.REAC_DIFF,
        multi_hot_atom_featurizer_mode: Literal["V1", "V2", "ORGANIC", "RIGR"] = "V2",
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = False,
        rxn_cols: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
        ignore_cols: Sequence[str] | None = None,
        weight_col: str | None = None,
        bounded: bool = False,
        no_header_row: bool = False,
    ):
        self.data_path = data_path
        self.reaction_mode = reaction_mode
        self.multi_hot_atom_featurizer_mode = multi_hot_atom_featurizer_mode
        self.keep_h = keep_h
        self.add_h = add_h
        self.ignore_stereo = ignore_stereo
        self.reorder_atoms = reorder_atoms
        self.rxn_cols = rxn_cols
        self.target_cols = target_cols
        self.ignore_cols = ignore_cols
        self.weight_col = weight_col
        self.bounded = bounded
        self.no_header_row = no_header_row

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X: Optional[Sequence[str]] = None, y=None, **__):
        return self._build_dataset(X, y)

    def transform(self, X: Optional[Sequence[str]] = None):
        return self._build_dataset(X, None)

    def _build_dataset(
        self,
        X: Optional[Sequence[str]],
        Y: Optional[Sequence[float]],
        weights=None,
        lt_mask=None,
        gt_mask=None,
    ) -> ReactionDataset:
        if self.data_path is not None:
            _, rxnss, Y, weights, lt_mask, gt_mask = parse_csv(
                path=self.data_path,
                smiles_cols=None,
                rxn_cols=self.rxn_cols,
                target_cols=self.target_cols,
                ignore_cols=self.ignore_cols,
                splits_cols=None,
                weight_col=self.weight_col,
                bounded=self.bounded,
                no_header_row=self.no_header_row,
            )
        else:
            if X is None:
                raise ValueError("No data supplied, X and data_path cannot both be None.")
            rxnss = [list(X)]
            weights, lt_mask, gt_mask = None, None, None
            if Y is None:
                Y = np.zeros((len(X), 1), dtype=float)
            elif Y.ndim == 1:
                Y = Y.reshape(-1, 1)

        _, rxn_data = make_datapoints(
            smiss=None,
            rxnss=rxnss,
            Y=Y,
            weights=weights,
            lt_mask=lt_mask,
            gt_mask=gt_mask,
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
            multi_hot_atom_featurizer_mode=self.multi_hot_atom_featurizer_mode,
        )


class ChempropMulticomponentTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        data_path: Optional[PathLike] = None,
        component_types: Sequence[Literal["molecule", "reaction"]] | None = None,
        reaction_mode="REAC_DIFF",
        multi_hot_atom_featurizer_mode: Literal["V1", "V2", "ORGANIC", "RIGR"] = "V2",
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = False,
        smiles_cols: Sequence[str] | None = None,
        rxn_cols: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
        ignore_cols: Sequence[str] | None = None,
        weight_col: str | None = None,
        bounded: bool = False,
        no_header_row: bool = False,
    ):
        self.data_path = data_path
        self.component_types = list(component_types or [])
        self.reaction_mode = reaction_mode
        self.multi_hot_atom_featurizer_mode = multi_hot_atom_featurizer_mode
        self.keep_h = keep_h
        self.add_h = add_h
        self.ignore_stereo = ignore_stereo
        self.reorder_atoms = reorder_atoms
        self.smiles_cols = smiles_cols
        self.rxn_cols = rxn_cols
        self.target_cols = target_cols
        self.ignore_cols = ignore_cols
        self.weight_col = weight_col
        self.bounded = bounded
        self.no_header_row = no_header_row
        self.mol_transformer = ChempropMoleculeTransformer(
            multi_hot_atom_featurizer_mode=multi_hot_atom_featurizer_mode,
            keep_h=keep_h,
            add_h=add_h,
            ignore_stereo=ignore_stereo,
            reorder_atoms=reorder_atoms,
        )
        self.rxn_transformer = ChempropReactionTransformer(
            multi_hot_atom_featurizer_mode=multi_hot_atom_featurizer_mode,
            reaction_mode=reaction_mode,
            keep_h=keep_h,
            add_h=add_h,
            ignore_stereo=ignore_stereo,
            reorder_atoms=reorder_atoms,
        )

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X: Optional[Sequence[str]] = None, y=None, **__):
        return self._build_dataset(X, y)

    def transform(self, X: Optional[Sequence[str]] = None):
        return self._build_dataset(X, None)

    def _build_dataset(self, X: Optional[Sequence[str]], Y):
        if self.data_path is not None:
            smiss, rxnss, Y, weights, lt_mask, gt_mask = parse_csv(
                path=self.data_path,
                smiles_cols=self.smiles_cols,
                rxn_cols=self.rxn_cols,
                target_cols=self.target_cols,
                ignore_cols=self.ignore_cols,
                splits_col=None,
                weight_col=self.weight_col,
                bounded=self.bounded,
                no_header_row=self.no_header_row,
            )
            datasets = [
                self.mol_transformer._build_dataset(
                    X=smiss, Y=Y, weights=weights, lt_mask=lt_mask, gt_mask=gt_mask
                ),
                self.rxn_transformer._build_dataset(
                    X=rxnss, Y=Y, weights=weights, lt_mask=lt_mask, gt_mask=gt_mask
                ),
            ]
        else:
            if X is None:
                raise ValueError("No data supplied, X and data_path cannot both be None.")
            if len(X) != len(self.component_types):
                logger.warning(
                    "data dimension and number of component_types inputted are inconsistent"
                )
            X = list(X)
            if Y is None:
                Y = np.zeros((len(X[0]), 1), dtype=float)
            elif Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            datasets = [
                self.mol_transformer._build_dataset(col, Y)
                if type == "molecule"
                else self.rxn_transformer._build_dataset(col, Y)
                for type, col in zip(self.component_types, X)
            ]
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
        num_workers: int = 0,
        batch_size: int = 64,
        output_dir: Optional[PathLike] = None,
        checkpoint: Optional[List[Path]] = None,
        molecule_featurizers: Optional[List[str]] = None,
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
        no_cache: bool = False,
        task_type: str = "regression",
        loss_function: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ):
        args = Namespace(
            num_workers=num_workers,
            batch_size=batch_size,
            output_dir=output_dir,
            checkpoint=checkpoint,
            molecule_featurizers=molecule_featurizers,
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
            no_cache=no_cache,
            task_type=task_type,
            loss_function=loss_function,
            metrics=metrics,
        )
        self.args = add_train_defaults(args)
        self.model = None
        patience = self.args.patience if self.args.patience is not None else self.args.epochs
        self.trainer = Trainer(
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            max_epochs=self.args.epochs,
            callbacks=[EarlyStopping(monitor="train_loss", patience=patience, mode="min")],
        )

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X: MoleculeDataset | ReactionDataset | MulticomponentDataset, y=None):
        if not self.args.no_cache:
            X.cache = True
        if self.args.checkpoint is not None:
            model_paths = find_models(self.args.checkpoint)
            if len(model_paths) != 1:
                logger.warning(
                    "More than one model path privided in checkpoint and only the first one is used. Call ChempropEnsembleRegressor instead."
                )
            model_path = model_paths[0]

            if isinstance(X, MulticomponentDataset):
                mpnn_cls = MulticomponentMPNN
            else:
                mpnn_cls = MPNN

            self.model = mpnn_cls.load_from_file(model_path)
            self.model.apply(
                lambda m: setattr(m, "p", self.args.dropout)
                if isinstance(m, torch.nn.Dropout)
                else None
            )
        else:
            input_transforms = normalize_inputs(X, X, self.args)
            output_transform = None
            if "regression" in self.args.task_type:
                output_scaler = X.normalize_targets()
                output_transform = UnscaleTransform.from_standard_scaler(output_scaler)
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
        if not self.args.no_cache:
            X.cache = True
        dl = DataLoader(
            X,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=pick_collate(X),
            drop_last=True,
        )
        preds = self.trainer.predict(self.model, dataloaders=dl)
        return torch.cat(preds, dim=0).view(-1).cpu().numpy()

    def score(self, X, y, metric: Literal["mae", "rmse", "mse", "r2", "accuracy"] = "rmse"):
        y_pred = self.predict(X)

        if metric == "mae":
            return mean_absolute_error(y, y_pred)
        elif metric == "rmse":
            return root_mean_squared_error(y, y_pred)
        elif metric == "mse":
            return root_mean_squared_error(y, y_pred) ** 2
        elif metric == "r2":
            return r2_score(y, y_pred)
        elif metric == "accuracy":
            return accuracy_score(y, (y_pred > 0.5).astype(int))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def _save_model(self):
        if self.args.model_output_dir is None:
            raise ValueError("model_output_dir is not specified")
        output_columns = self.args.target_columns
        save_model(self.args.model_output_dir / "best.pt", self.model, output_columns)
        return self


class ChempropEnsembleRegressor(ChempropRegressor):
    def __init__(self, ensemble_size: int = 5, **chemprop_kwargs):
        self.ensemble_size = ensemble_size
        self.base_kwargs = chemprop_kwargs
        super().__init__(**chemprop_kwargs)
        self.models: List[ChempropRegressor] = []

    def __sklearn_is_fitted__(self):
        return len(self.models) == self.ensemble_size

    def fit(self, X, y):
        self.models = []
        if self.args.checkpoint is not None:
            if len(self.args.checkpoint) != self.ensemble_size:
                logger.warning(
                    f"The number of models in ensemble for each splitting of data is set to {len(self.args.checkpoint)}."
                )
                self.ensemble_size = len(self.args.checkpoint)

            for path in self.args.checkpoint:
                args = dict(self.base_kwargs)
                args["checkpoint"] = [path]
                model = ChempropRegressor(**args)
                model.fit(X, y)
                self.models.append(model)
        else:
            for _ in range(self.ensemble_size):
                model = ChempropRegressor(**self.base_kwargs)
                model.fit(X, y)
                self.models.append(model)
        return self

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.mean(preds, axis=0)

    def score(self, X, y, metric: Literal["mae", "rmse", "mse", "r2", "accuracy"] = "rmse"):
        y_pred = self.predict(X)

        if metric == "mae":
            return mean_absolute_error(y, y_pred)
        elif metric == "rmse":
            return root_mean_squared_error(y, y_pred)
        elif metric == "mse":
            return root_mean_squared_error(y, y_pred) ** 2
        elif metric == "r2":
            return r2_score(y, y_pred)
        elif metric == "accuracy":
            return accuracy_score(y, (y_pred > 0.5).astype(int))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def _save_models(self):
        if self.args.model_output_dir is None:
            raise ValueError("no output directory specified")
        output_columns = self.args.target_columns
        for idx, model in enumerate(self.models):
            save_model(self.args.model_output_dir / f"best_{idx}.pt", model, output_columns)
        return self


if __name__ == "__main__":
    # microtest
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, train_test_split
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    sklearnPipeline = Pipeline(
        [
            (
                "featurizer",
                ChempropMulticomponentTransformer(
                    data_path="rxn+mol.csv",
                    smiles_cols="solvent_smiles",
                    rxn_cols="rxn_smiles",
                    target_cols="target",
                ),
            ),
            ("regressor", ChempropRegressor()),
        ]
    )

    # df = pd.read_csv("rxn.csv")  # change to target datapath
    # X =
    # df["smiles"].to_numpy(dtype=str)
    # y = df["ea"].to_numpy(dtype=float)

    df = pd.read_csv("rxn+mol.csv")  # change to target datapath
    X = (df["rxn_smiles"].to_numpy(dtype=str), df["solvent_smiles"].to_numpy(dtype=str))
    y = df["target"].to_numpy(dtype=float)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    sklearnPipeline.fit(X, y)
    score = sklearnPipeline.score(X, y)
    print(f"RMSE: {score}")

    # scores = cross_val_score(sklearnPipeline, X, y, cv=5, scoring="neg_mean_squared_error")
    # print("Cross-validation scores:", scores)
    # print("Mean MSE:", -scores.mean())
