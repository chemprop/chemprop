from argparse import ArgumentParser, Namespace
from datetime import datetime
import logging
import os
from os import PathLike
from pathlib import Path
from typing import List, Literal, Optional, Sequence

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, root_mean_squared_error
import torch
from torch.utils.data import DataLoader

from chemprop.cli.common import add_common_args, find_models
from chemprop.cli.train import add_train_args, build_model, normalize_inputs
from chemprop.cli.utils.parsing import make_datapoints, make_dataset, parse_csv
from chemprop.data.collate import collate_batch, collate_mol_atom_bond_batch, collate_multicomponent
from chemprop.data.datasets import MolAtomBondDataset, MulticomponentDataset
from chemprop.featurizers.molgraph.reaction import RxnMode
from chemprop.models import MPNN, MulticomponentMPNN, utils
from chemprop.nn.transforms import UnscaleTransform

logger = logging.getLogger(__name__)

NOW = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
CHEMPROP_TRAIN_DIR = Path(os.getenv("CHEMPROP_TRAIN_DIR", "chemprop_training"))


class ChempropMoleculeTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = False,
        smiles_cols: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
        ignore_cols: Sequence[str] | None = None,
        weight_col: str | None = None,
        bounded: bool = False,
        no_header_row: bool = False,
    ):
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

    def fit_transform(self, X: np.ndarray | Sequence[str] | PathLike, y=None, **_):
        return self._build_dps(X, y)

    def transform(self, X: np.ndarray | Sequence[str] | PathLike):
        return self._build_dps(X, None)

    def _build_dps(
        self,
        X: np.ndarray | Sequence[str] | PathLike,
        Y: Optional[np.ndarray],
        weights=None,
        lt_mask=None,
        gt_mask=None,
    ):
        if isinstance(X, PathLike):
            smiss, _, Y, weights, lt_mask, gt_mask, _ = parse_csv(
                path=X,
                smiles_cols=self.smiles_cols,
                rxn_cols=None,
                target_cols=self.target_cols,
                ignore_cols=self.ignore_cols,
                splits_col=None,
                weight_col=self.weight_col,
                descriptor_cols=None,
                bounded=self.bounded,
                no_header_row=self.no_header_row,
            )
        else:
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
            use_cuikmolmaker_featurization=False,
        )
        return mol_data[0]


class ChempropReactionTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
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

    def fit_transform(self, X: np.ndarray | Sequence[str] | PathLike, y=None, **__):
        return self._build_dps(X, y)

    def transform(self, X: np.ndarray | Sequence[str] | PathLike):
        return self._build_dps(X, None)

    def _build_dps(
        self,
        X: np.ndarray | Sequence[str] | PathLike,
        Y: Optional[Sequence[float]],
        weights=None,
        lt_mask=None,
        gt_mask=None,
    ):
        if isinstance(X, PathLike):
            _, rxnss, Y, weights, lt_mask, gt_mask, _ = parse_csv(
                path=X,
                smiles_cols=None,
                rxn_cols=self.rxn_cols,
                target_cols=self.target_cols,
                ignore_cols=self.ignore_cols,
                splits_col=None,
                weight_col=self.weight_col,
                descriptor_cols=None,
                bounded=self.bounded,
                no_header_row=self.no_header_row,
            )
        else:
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
            use_cuikmolmaker_featurization=False,
        )
        return rxn_data[0]


class ChempropMulticomponentTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        component_types: Sequence[Literal["molecule", "reaction"]] | None = None,
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
        self.component_types = component_types
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
            keep_h=keep_h, add_h=add_h, ignore_stereo=ignore_stereo, reorder_atoms=reorder_atoms
        )
        self.rxn_transformer = ChempropReactionTransformer(
            keep_h=keep_h, add_h=add_h, ignore_stereo=ignore_stereo, reorder_atoms=reorder_atoms
        )

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X: np.ndarray | Sequence[Sequence[str]] | PathLike, y=None, **__):
        return self._build_dps(X, y)

    def transform(self, X: np.ndarray | Sequence[Sequence[str]] | PathLike):
        return self._build_dps(X, None)

    def _build_dps(
        self, X: np.ndarray | Sequence[Sequence[str]] | PathLike, Y: Optional[Sequence[float]]
    ):
        if isinstance(X, PathLike):
            smiss, rxnss, Y, weights, lt_mask, gt_mask, _ = parse_csv(
                path=X,
                smiles_cols=self.smiles_cols,
                rxn_cols=self.rxn_cols,
                target_cols=self.target_cols,
                ignore_cols=self.ignore_cols,
                splits_col=None,
                weight_col=self.weight_col,
                descriptor_cols=None,
                bounded=self.bounded,
                no_header_row=self.no_header_row,
            )
            mol_dps = self.mol_transformer._build_dps(
                X=smiss, Y=Y, weights=weights, lt_mask=lt_mask, gt_mask=gt_mask
            )
            rxn_dps = self.rxn_transformer._build_dps(
                X=rxnss, Y=Y, weights=weights, lt_mask=lt_mask, gt_mask=gt_mask
            )
            return [mol_dps, rxn_dps]
        else:
            if (
                isinstance(X, (np.ndarray, list))
                and np.ndim(X) == 2
                and np.asarray(X).shape[1] == len(self.component_types)
            ):
                X = np.asarray(X).T.tolist()
            else:
                X = list(X)
            if len(X) != len(self.component_types):
                logger.warning(
                    "data dimension and number of component_types inputted are inconsistent"
                )
            if Y is None:
                Y = np.zeros((len(X[0]), 1), dtype=float)
            elif np.ndim(Y) == 1:
                Y = np.asarray(Y).reshape(-1, 1)
            dp_lists = [
                self.mol_transformer._build_dps(col, Y)
                if type == "molecule"
                else self.rxn_transformer._build_dps(col, Y)
                for type, col in zip(self.component_types, X)
            ]
            return dp_lists


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


def _split_indices(n: int, val_size: float):
    if val_size <= 0:
        return np.arange(n), np.array([], dtype=int)
    if val_size >= 1:
        raise ValueError("val_size should be a float less than 1")
    n_val = int(n * val_size)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


class ChempropRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 64,
        output_dir: Optional[PathLike] = CHEMPROP_TRAIN_DIR / "sklearn_output" / NOW,
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
        val_size: float = 0,
        multi_hot_atom_featurizer_mode: Literal["V1", "V2", "ORGANIC", "RIGR"] = "V2",
        reaction_mode: RxnMode = RxnMode.REAC_DIFF,
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
            val_size=val_size,
            multi_hot_atom_featurizer_mode=multi_hot_atom_featurizer_mode,
            reaction_mode=reaction_mode,
        )
        self.args = add_train_defaults(args)
        self.model = None
        for name, value in locals().items():
            if name not in {"self", "args"}:
                setattr(self, name, value)

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        if self.args.checkpoint is not None:
            model_paths = find_models(self.args.checkpoint)
            if len(model_paths) != 1:
                logger.warning(
                    "More than one model path provided in checkpoint and only the first one is used. Call ChempropEnsembleRegressor instead."
                )
            model_path = model_paths[0]

            if isinstance(X[0], list):
                mpnn_cls = MulticomponentMPNN
            else:
                mpnn_cls = MPNN

            self.model = mpnn_cls.load_from_file(model_path)
            self.model.apply(
                lambda m: setattr(m, "p", self.args.dropout)
                if isinstance(m, torch.nn.Dropout)
                else None
            )

        if isinstance(X[0], list):
            n = len(X[0])
            train_idx, val_idx = _split_indices(n, self.args.val_size)

            train_datasets = []
            val_datasets = []

            for dp_list in X:
                train_dps = [dp_list[i] for i in train_idx]
                val_dps = [dp_list[i] for i in val_idx]

                train_ds = make_dataset(
                    train_dps,
                    reaction_mode=self.reaction_mode,
                    multi_hot_atom_featurizer_mode=self.multi_hot_atom_featurizer_mode,
                )
                train_datasets.append(train_ds)

                if len(val_dps) > 0:
                    val_ds = make_dataset(
                        val_dps,
                        reaction_mode=self.reaction_mode,
                        multi_hot_atom_featurizer_mode=self.multi_hot_atom_featurizer_mode,
                    )
                    val_datasets.append(val_ds)

            train_set = MulticomponentDataset(train_datasets)
            val_set = MulticomponentDataset(val_datasets) if len(val_datasets) > 0 else None

        else:
            if not isinstance(X, (list, tuple)):
                raise ValueError("X must be a list of datapoints for non-multicomponent inputs")
            n = len(X)
            train_idx, val_idx = _split_indices(n, self.args.val_size)
            train_dps = [X[i] for i in train_idx]
            val_dps = [X[i] for i in val_idx]
            train_set = make_dataset(
                train_dps,
                reaction_mode=self.reaction_mode,
                multi_hot_atom_featurizer_mode=self.multi_hot_atom_featurizer_mode,
            )
            val_set = None
            if len(val_dps) > 0:
                val_set = make_dataset(
                    val_dps,
                    reaction_mode=self.reaction_mode,
                    multi_hot_atom_featurizer_mode=self.multi_hot_atom_featurizer_mode,
                )

        if not self.args.no_cache:
            train_set.cache = True
            if val_set is not None:
                val_set.cache = True

        if self.model is None:
            input_transforms = normalize_inputs(
                train_set, val_set if val_set is not None else train_set, self.args
            )
            output_transform = None
            if "regression" in self.args.task_type:
                output_scaler = train_set.normalize_targets()
                output_transform = UnscaleTransform.from_standard_scaler(output_scaler)
            self.model = build_model(self.args, train_set, output_transform, input_transforms)

        train_loader = DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=pick_collate(train_set),
        )
        if val_set:
            val_loader = DataLoader(
                val_set,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=pick_collate(val_set),
            )
        else:
            val_loader = None

        patience = self.args.patience if self.args.patience else self.args.epochs
        metric = "val_loss" if val_loader else "train_loss"
        trainer = Trainer(
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            max_epochs=self.args.epochs,
            callbacks=[EarlyStopping(monitor=metric, patience=patience, mode="min")],
        )

        if val_loader:
            trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            trainer.fit(self.model, train_dataloaders=train_loader)

        return self

    def predict(self, X):
        if isinstance(X[0], list):
            test_set = MulticomponentDataset(
                [
                    make_dataset(
                        dps,
                        reaction_mode=self.reaction_mode,
                        multi_hot_atom_featurizer_mode=self.multi_hot_atom_featurizer_mode,
                    )
                    for dps in X
                ]
            )
            self._y = test_set.datasets[0].Y
        else:
            test_set = make_dataset(
                X,
                reaction_mode=self.reaction_mode,
                multi_hot_atom_featurizer_mode=self.multi_hot_atom_featurizer_mode,
            )
            self._y = test_set.Y
        if not self.args.no_cache:
            test_set.cache = True
        dl = DataLoader(
            test_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=pick_collate(test_set),
        )
        eval_trainer = Trainer(
            accelerator=self.args.accelerator, devices=1, enable_progress_bar=True
        )
        preds = eval_trainer.predict(self.model, dataloaders=dl, return_predictions=True)
        return torch.cat(preds, dim=0).view(-1).cpu().numpy()

    def score(self, X, y=None, metric: Literal["mae", "rmse", "mse", "r2", "accuracy"] = "rmse"):
        y_pred = self.predict(X)
        if y is None:
            y = self._y
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

    def save_model(self, output_dir: Optional[PathLike] = None):
        if output_dir is None:
            output_dir = self.args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        utils.save_model(str(output_dir / "best.pt"), self.model, None)
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
        if y is None:
            y = self.models[0]._y
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

    def save_model(self, output_dir: Optional[PathLike] = None):
        if output_dir is None:
            output_dir = self.args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, model in enumerate(self.models):
            utils.save_model(str(output_dir / f"best_{idx}.pt"), model.model, None)
        return self


# microtest
if __name__ == "__main__":
    from sklearn.pipeline import Pipeline

    sklearnPipeline = Pipeline(
        [
            (
                "featurizer",
                ChempropMulticomponentTransformer(
                    smiles_cols="solvent_smiles", rxn_cols="rxn_smiles", target_cols="target"
                ),
            ),
            ("regressor", ChempropRegressor(epochs=100, patience=10)),
        ]
    )

    sklearnPipeline.fit(X=Path("tests/data/regression/rxn+mol/rxn+mol.csv"))
    score = sklearnPipeline.score(X=Path("tests/data/regression/rxn+mol/rxn+mol.csv"))
    print(f"RMSE: {score}")
