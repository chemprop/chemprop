import logging
from os import PathLike
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chemprop.data.datapoints import MoleculeDatapoint, ReactionDatapoint
from chemprop.data.datasets import MoleculeDataset, ReactionDataset
from chemprop.featurizers.molecule import MoleculeFeaturizer
from chemprop.featurizers.molgraph import (
    CondensedGraphOfReactionFeaturizer,
    SimpleMoleculeMolGraphFeaturizer,
)
from chemprop.featurizers.atom import get_multi_hot_atom_featurizer

logger = logging.getLogger(__name__)


def parse_csv(
    path: PathLike,
    smiles_cols: Sequence[str] | None,
    rxn_cols: Sequence[str] | None,
    target_cols: Sequence[str] | None,
    ignore_cols: Sequence[str] | None,
    weight_col: str | None,
    bounded: bool = False,
    no_header_row: bool = False,
):
    df = pd.read_csv(path, header=None if no_header_row else "infer", index_col=False)

    if smiles_cols is not None and rxn_cols is not None:
        smiss = df[smiles_cols].values.tolist()
        rxnss = df[rxn_cols].values.tolist()
        input_cols = [*smiles_cols, *rxn_cols]
    elif smiles_cols is not None and rxn_cols is None:
        smiss = df[smiles_cols].values.tolist()
        rxnss = None
        input_cols = smiles_cols
    elif smiles_cols is None and rxn_cols is not None:
        smiss = None
        rxnss = df[rxn_cols].values.tolist()
        input_cols = rxn_cols
    else:
        smiss = df.iloc[:, [0]].values.tolist()
        rxnss = None
        input_cols = [df.columns[0]]

    if target_cols is None:
        target_cols = list(set(df.columns) - set(input_cols) - set(ignore_cols or []))

    Y = df[target_cols]
    weights = None if weight_col is None else df[weight_col].to_numpy()

    if bounded:
        lt_mask = Y.applymap(lambda x: "<" in x).to_numpy()
        gt_mask = Y.applymap(lambda x: ">" in x).to_numpy()
        Y = Y.applymap(lambda x: x.strip("<").strip(">")).astype(float).to_numpy()
    else:
        Y = Y.to_numpy()
        lt_mask = None
        gt_mask = None

    return smiss, rxnss, Y, weights, lt_mask, gt_mask


def get_column_names(
    path: PathLike,
    smiles_cols: Sequence[str] | None,
    rxn_cols: Sequence[str] | None,
    target_cols: Sequence[str] | None,
    ignore_cols: Sequence[str] | None,
    no_header_row: bool = False,
):
    df = pd.read_csv(path, header=None if no_header_row else "infer", index_col=False)

    input_cols = []
    target_cols = []

    if smiles_cols is not None:
        input_cols.extend(smiles_cols)
    if rxn_cols is not None:
        input_cols.extend(rxn_cols)
    if target_cols is not None:
        target_cols.extend(target_cols)

    if len(input_cols) == 0:
        if no_header_row:
            input_cols = ["SMILES"]
        else:
            input_cols = [df.columns[0]]

    if len(target_cols) == 0:
        if no_header_row:
            ignore_len = len(ignore_cols) if ignore_cols else 0
            ["pred_" + str(i) for i in range((len(df.columns) - len(input_cols) - ignore_len))]
        else:
            target_cols = list(set(df.columns) - set(input_cols) - set(ignore_cols or []))

    cols = input_cols + target_cols
    return cols


def make_datapoints(
    smiss: list[list[str]] | None,
    rxnss: list[list[str]] | None,
    Y: np.ndarray,
    weights: np.ndarray | None,
    lt_mask: np.ndarray | None,
    gt_mask: np.ndarray | None,
    X_d: np.ndarray | None,
    V_fs: list[np.ndarray] | None,
    E_fs: list[np.ndarray] | None,
    V_ds: list[np.ndarray] | None,
    features_generators: list[MoleculeFeaturizer] | None,
    keep_h: bool,
    add_h: bool,
) -> tuple[list[list[MoleculeDatapoint]], list[list[ReactionDatapoint]]]:
    """Make the :class:`MoleculeDatapoint`s and :class:`ReactionDatapoint`s for a given
    dataset.
    Parameters
    ----------

    Returns
    -------
    list[list[MoleculeDatapoint]]
        a list of lists of :class:`MoleculeDatapoint`s of shape ``j x n``, where ``j`` is the
        number of molecule components per datapoint and ``n`` is the total number of datapoints
    list[list[ReactionDatapoint]]
        a list of lists of :class:`ReactionDatapoint`s of shape ``k x n``, where ``k`` is the
        number of reaction components per datapoint and ``n`` is the total number of datapoints
    .. note::
        either ``j`` or ``k`` may be 0, in which case the corresponding list will be empty.
    """
    if smiss is None and rxnss is None:
        raise ValueError("args 'smiss' and 'rnxss' were both `None`!")
    elif rxnss is None:
        N = len(smiss)
        rxnss = []
    elif smiss is None:
        N = len(rxnss)
        smiss = []
    elif len(smiss) != len(rxnss):
        raise ValueError(
            f"args 'smiss' and 'rxnss' must have same length! got {len(smiss)} and {len(rxnss)}"
        )
    else:
        N = len(smiss)

    weights = np.ones(N) if weights is None else weights
    gt_mask = [None] * N if gt_mask is None else gt_mask
    lt_mask = [None] * N if lt_mask is None else lt_mask

    n_mols = len(smiss)
    X_d = [None] * N if X_d is None else X_d
    V_fs = [[None] * N] * n_mols if V_fs is None else V_fs
    E_fs = [[None] * N] * n_mols if E_fs is None else E_fs
    V_ds = [[None] * N] * n_mols if V_ds is None else V_ds

    mol_data = [
        [
            MoleculeDatapoint.from_smi(
                smis[i],
                keep_h=keep_h,
                add_h=add_h,
                y=Y[i],
                weight=weights[i],
                gt_mask=gt_mask[i],
                lt_mask=lt_mask[i],
                x_d=X_d[i],
                mfs=features_generators,
                x_phase=None,
                V_f=V_fs[mol_idx][i],
                E_f=E_fs[mol_idx][i],
                V_d=V_ds[mol_idx][i],
            )
            for i in range(N)
        ]
        for mol_idx, smis in enumerate(zip(*smiss))
    ]
    rxn_data = [
        [
            ReactionDatapoint.from_smi(
                rxns[i],
                keep_h=keep_h,
                add_h=add_h,
                y=Y[i],
                weight=weights[i],
                gt_mask=gt_mask[i],
                lt_mask=lt_mask[i],
                x_d=X_d[i],
                mfs=features_generators,
                x_phase=None,
            )
            for i in range(N)
        ]
        for rxn_idx, rxns in enumerate(zip(*rxnss))
    ]

    return mol_data, rxn_data


def build_data_from_files(
    p_data: PathLike,
    no_header_row: bool,
    smiles_cols: Sequence[str] | None,
    rxn_cols: Sequence[str] | None,
    target_cols: Sequence[str] | None,
    ignore_cols: Sequence[str] | None,
    weight_col: str | None,
    bounded: bool,
    p_descriptors: PathLike,
    p_atom_feats: PathLike,
    p_bond_feats: PathLike,
    p_atom_descs: PathLike,
    **featurization_kwargs: Mapping,
) -> list[list[MoleculeDatapoint] | list[ReactionDatapoint]]:
    smiss, rxnss, Y, weights, lt_mask, gt_mask = parse_csv(
        p_data, smiles_cols, rxn_cols, target_cols, ignore_cols, weight_col, bounded, no_header_row
    )
    n_molecules = len(list(zip(*smiss))) if smiss is not None else 0

    X_ds = load_input_feats_and_descs(p_descriptors, None, feat_desc="X_d")
    V_fss = load_input_feats_and_descs(p_atom_feats, n_molecules, feat_desc="V_f")
    E_fss = load_input_feats_and_descs(p_bond_feats, n_molecules, feat_desc="E_f")
    V_dss = load_input_feats_and_descs(p_atom_descs, n_molecules, feat_desc="V_d")

    mol_data, rxn_data = make_datapoints(
        smiss,
        rxnss,
        Y,
        weights,
        lt_mask,
        gt_mask,
        X_ds,
        V_fss,
        E_fss,
        V_dss,
        **featurization_kwargs,
    )

    return mol_data + rxn_data


def load_input_feats_and_descs(paths, n_molecules, feat_desc):
    if paths is None:
        return None

    match feat_desc:
        case "X_d":
            path = paths
            loaded_feature = np.load(path)
            features = loaded_feature["arr_0"]

        case _:
            features = []
            for _ in range(n_molecules):
                path = paths  # TODO: currently only supports a single path
                loaded_feature = np.load(path)
                loaded_feature = [loaded_feature[f"arr_{i}"] for i in range(len(loaded_feature))]
                features.append(loaded_feature)
    return features


def make_dataset(
    data: Sequence[MoleculeDatapoint] | Sequence[ReactionDatapoint],
    reaction_mode: str,
    multi_hot_atom_featurizer_mode: str = "V2",
) -> MoleculeDataset | ReactionDataset:

    atom_featurizer = get_multi_hot_atom_featurizer(multi_hot_atom_featurizer_mode)

    if isinstance(data[0], MoleculeDatapoint):
        extra_atom_fdim = data[0].V_f.shape[1] if data[0].V_f is not None else 0
        extra_bond_fdim = data[0].E_f.shape[1] if data[0].E_f is not None else 0
        featurizer = SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=atom_featurizer,
            extra_atom_fdim=extra_atom_fdim,
            extra_bond_fdim=extra_bond_fdim,
        )
        return MoleculeDataset(data, featurizer)

    featurizer = CondensedGraphOfReactionFeaturizer(
        mode_=reaction_mode, atom_featurizer=atom_featurizer
    )

    return ReactionDataset(data, featurizer)
