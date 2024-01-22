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


def make_datapoints(
    smiss: list[list[str]] | None,
    rxnss: list[list[str]] | None,
    Y: np.ndarray,
    weights: np.ndarray | None,
    lt_mask: np.ndarray | None,
    gt_mask: np.ndarray | None,
    X_f: np.ndarray | None,
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
    X_f = [None] * N if X_f is None else X_f
    V_fs = [None] * N if V_fs is None else V_fs
    E_fs = [None] * N if E_fs is None else E_fs
    V_ds = [None] * N if V_ds is None else V_ds

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
                x_f=X_f[i],
                mfs=features_generators,
                x_phase=None,
                V_f=V_fs[i],
                E_f=E_fs[i],
                V_d=V_ds[i],
            )
            for i in range(N)
        ]
        for smis in list(zip(*smiss))
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
                x_f=X_f[i],
                mfs=features_generators,
                x_phase=None,
            )
            for i in range(N)
        ]
        for rxns in list(zip(*rxnss))
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
    p_features: PathLike,
    p_atom_feats: PathLike,
    p_bond_feats: PathLike,
    p_atom_descs: PathLike,
    **featurization_kwargs: Mapping,
) -> list[MoleculeDatapoint] | list[ReactionDatapoint]:
    smiss, rxnss, Y, weights, lt_mask, gt_mask = parse_csv(
        p_data, smiles_cols, rxn_cols, target_cols, ignore_cols, weight_col, bounded, no_header_row
    )
    X_f = np.load(p_features) if p_features else None
    V_fs = np.load(p_atom_feats, allow_pickle=True) if p_atom_feats else None
    E_fs = np.load(p_bond_feats, allow_pickle=True) if p_bond_feats else None
    V_ds = np.load(p_atom_descs, allow_pickle=True) if p_atom_descs else None

    mol_data, rxn_data = make_datapoints(
        smiss, rxnss, Y, weights, lt_mask, gt_mask, X_f, V_fs, E_fs, V_ds, **featurization_kwargs
    )

    # NOTE: return only a single component for now with a preference for rxns
    data = rxn_data if len(rxn_data) > 0 else mol_data
    return data


def make_dataset(
    data: Sequence[MoleculeDatapoint] | Sequence[ReactionDatapoint], reaction_mode: str
) -> MoleculeDataset | ReactionDataset:
    if isinstance(data[0], MoleculeDatapoint):
        extra_atom_fdim = data[0].V_f.shape[1] if data[0].V_f is not None else 0
        extra_bond_fdim = data[0].E_f.shape[1] if data[0].E_f is not None else 0
        featurizer = SimpleMoleculeMolGraphFeaturizer(
            extra_atom_fdim=extra_atom_fdim, extra_bond_fdim=extra_bond_fdim
        )
        return MoleculeDataset(data, featurizer)

    featurizer = CondensedGraphOfReactionFeaturizer(mode_=reaction_mode)

    return ReactionDataset(data, featurizer)
