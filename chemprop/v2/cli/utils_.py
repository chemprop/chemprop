import csv
import logging
import pandas as pd
from os import PathLike
from typing import Mapping, Optional, Sequence, Type

import numpy as np

from chemprop.v2 import models
from chemprop.v2.data.datapoints import MoleculeDatapoint, _DatapointMixin, ReactionDatapoint
from chemprop.v2.data.datasets import MoleculeDataset, ReactionDataset
from chemprop.v2.featurizers.molecule import MoleculeFeaturizerRegistry
from chemprop.v2.featurizers.molgraph import (
    MoleculeMolGraphFeaturizerProto,
    CondensedGraphOfReactionFeaturizer,
)


logger = logging.getLogger(__name__)


def optional_float(x: str) -> float:
    return float(x) if x else float("nan")


def parse_data_csv(
    path: PathLike,
    no_header_row: bool = False,
    smiles_cols: Optional[Sequence[int]] = None,
    target_cols: Optional[Sequence[int]] = None,
    bounded: bool = False,
):
    smiles_cols = smiles_cols or [0]

    with open(path) as fid:
        reader = csv.reader(fid)

        if not no_header_row:
            header = next(reader)
            if target_cols is None:
                target_cols = [i for i in range(header) if i not in smiles_cols]
            # smiles_names = [header[i] for i in smiles_cols]
            task_names = [header[i] for i in target_cols]
            logger.info(f"Parsed tasks: {task_names}")
        else:
            target_cols = [1] if target_cols is None else target_cols
            task_names = [f"task_{i}" for i in target_cols]
            # smiles_names = [f"smiles_{i}" for i in smiles_cols]

        logger.info(f"Parsed {len(task_names)} targets from {path}")

        smiss = []
        raw_targetss = []
        for i, row in enumerate(reader):
            try:
                smis = [row[j] for j in smiles_cols]
                targets = [row[j] for j in target_cols]
            except IndexError as e:
                raise ValueError(
                    f"Input data file contained a ragged row! The culprit is L{i} of {path}!"
                )

            smiss.append(smis)
            raw_targetss.append(targets)

        try:
            targetss = []

            if not bounded:
                gt_targetss = None
                lt_targetss = None
                for i, raw_targets in enumerate(raw_targetss):
                    targetss.append([optional_float(t) for t in raw_targets])
            else:
                gt_targetss = []
                lt_targetss = []
                for i, raw_targets in enumerate(raw_targetss):
                    targets = []
                    for t in raw_targets:
                        pass
                    targets, gt_targets, lt_targets = zip(*[bounded_float(t) for t in raw_targets])
                    targetss.append(targets)
                    gt_targetss.append(gt_targets)
                    lt_targetss.append(lt_targets)
                gt_targetss = np.array(gt_targetss, bool)
                lt_targetss = np.array(lt_targetss, bool)
        except ValueError as e:
            raise ValueError(
                f"Bad target formatting! The culprit is L{i} of {path}! "
                f"See message for more details: {e.message}"
            )

        targetss = np.array(targetss, float)

    logger.debug(f"{targetss.shape[0]} molecules | {targetss.shape[1]} tasks")
    logger.debug(f"{np.isfinite(targetss).sum()}/{targetss.size} valid targets")

    return smiss, targetss, gt_targetss, lt_targetss


# NOTE(degraff): this should be generalized in the future to accept `smis: list[list[str]]` and
# return `list[CompositeDatapoint]` for models using a `CompositeMessagePassingBlock`
def make_datapoints(
    smis: list[str],
    targetss: np.ndarray,
    weights: np.ndarray | None,
    gt_targetss: np.ndarray | None,
    lt_targetss: np.ndarray | None,
    featuress: np.ndarray | None,
    atom_features: np.ndarray | None,
    bond_features: np.ndarray | None,
    atom_descriptors: np.ndarray | None,
    features_generators: Optional[str],
    keep_h: bool,
    add_h: bool,
    reaction: bool,
) -> list[_DatapointMixin]:
    weights = np.ones(len(smis)) if weights is None else weights
    gt_targetss = [None] * len(smis) if gt_targetss is None else gt_targetss
    lt_targetss = [None] * len(smis) if lt_targetss is None else lt_targetss
    featuress = [None] * len(smis) if featuress is None else featuress
    mfs = [MoleculeFeaturizerRegistry.get(features_generators)()] if features_generators else None

    if reaction:
        data = [
            ReactionDatapoint.from_smi(
                smis[i],
                keep_h,
                add_h,
                targetss[i],
                weights[i],
                gt_targetss[i],
                lt_targetss[i],
                featuress[i],
                mfs,
                None,
            )
            for i in range(len(smis))
        ]
    else:
        if atom_features is None:
            atom_features = [None] * (len(smis))

        if bond_features is None:
            bond_features = [None] * (len(smis))

        if atom_descriptors is None:
            atom_descriptors = [None] * (len(smis))

        data = [
            MoleculeDatapoint.from_smi(
                smis[i],
                targetss[i],
                weights[i],
                gt_targetss[i],
                lt_targetss[i],
                featuress[i],
                mfs,
                None,
                keep_h,
                add_h,
                atom_features[i],
                bond_features[i],
                atom_descriptors[i],
            )
            for i in range(len(smis))
        ]

    return data


def build_data_from_files(
    p_data: PathLike,
    no_header_row,
    smiles_columns,
    target_columns,
    bounded,
    p_features: PathLike,
    p_atom_feats: PathLike,
    p_bond_feats: PathLike,
    p_atom_descs: PathLike,
    data_weights_path: PathLike,
    **featurization_kwargs: Mapping,
) -> list[_DatapointMixin]:
    smiss, targetss, gt_targetss, lt_targetss = parse_data_csv(
        p_data, no_header_row, smiles_columns, target_columns, bounded
    )
    featuress = np.load(p_features) if p_features else None
    atom_featss = np.load(p_atom_feats, allow_pickle=True) if p_atom_feats else None
    bond_featss = np.load(p_bond_feats, allow_pickle=True) if p_bond_feats else None
    atom_descss = np.load(p_atom_descs, allow_pickle=True) if p_atom_descs else None
    weights = pd.read_csv(data_weights_path, header=None).values if data_weights_path else None

    smis = [smis[0] for smis in smiss]  # only use 0th input for now
    data = make_datapoints(
        smis,
        targetss,
        weights,
        gt_targetss,
        lt_targetss,
        featuress,
        atom_featss,
        bond_featss,
        atom_descss,
        **featurization_kwargs,
    )

    return data


def make_dataset(
    data: Sequence[_DatapointMixin], bond_messages: bool, reaction_mode: str
) -> MoleculeDataset | ReactionDataset:
    if isinstance(data[0], MoleculeDatapoint):
        extra_atom_fdim = data[0].V_f.shape[1] if data[0].V_f is not None else 0
        extra_bond_fdim = data[0].E_f.shape[1] if data[0].E_f is not None else 0
        featurizer = MoleculeMolGraphFeaturizerProto(
            bond_messages=bond_messages,
            extra_atom_fdim=extra_atom_fdim,
            extra_bond_fdim=extra_bond_fdim,
        )
        return MoleculeDataset(data, featurizer)

    featurizer = CondensedGraphOfReactionFeaturizer(
        bond_messages=bond_messages, mode_=reaction_mode
    )

    return ReactionDataset(data, featurizer)


def get_mpnn_cls(task_type: str, loss_function: Optional[str] = None) -> Type[models.MPNN]:
    if task_type == "regression":
        if loss_function == "mve":
            return models.MveRegressionMPNN
        elif loss_function == "evidential":
            return models.EvidentialMPNN
        elif loss_function in ("bounded", "mse", None):
            return models.RegressionMPNN
    elif task_type == "classification":
        if loss_function == "dirichlet":
            return models.DirichletClassificationMPNN
        elif loss_function in ("mcc", "bce", None):
            return models.BinaryClassificationMPNN
    elif task_type == "multiclass":
        if loss_function == "dirichlet":
            return models.DirichletMulticlassMPNN
        elif loss_function in ("mcc", "ce", None):
            return models.MulticlassMPNN
    elif task_type == "spectral":
        if loss_function in ("sid", "wasserstein", None):
            return models.SpectralMPNN

    raise ValueError(
        f"Incompatible dataset type ('{task_type}') and loss function ('{loss_function}')!"
    )
