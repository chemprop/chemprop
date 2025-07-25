import logging
from os import PathLike
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from torch import nn

from chemprop.data.datapoints import (
    LazyMoleculeDatapoint,
    MolAtomBondDatapoint,
    MoleculeDatapoint,
    ReactionDatapoint,
)
from chemprop.data.datasets import (
    CuikmolmakerDataset,
    MolAtomBondDataset,
    MoleculeDataset,
    ReactionDataset,
)
from chemprop.featurizers.atom import get_multi_hot_atom_featurizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer, RIGRBondFeaturizer
from chemprop.featurizers.molecule import MoleculeFeaturizerRegistry
from chemprop.featurizers.molgraph import (
    CondensedGraphOfReactionFeaturizer,
    CuikmolmakerMolGraphFeaturizer,
    RxnMode,
    SimpleMoleculeMolGraphFeaturizer,
)
from chemprop.utils import make_mol

logger = logging.getLogger(__name__)


def parse_csv(
    path: PathLike,
    smiles_cols: Sequence[str] | None,
    rxn_cols: Sequence[str] | None,
    target_cols: Sequence[str] | None,
    ignore_cols: Sequence[str] | None,
    splits_col: str | None,
    weight_col: str | None,
    descriptor_cols: Sequence[str] | None,
    bounded: bool = False,
    no_header_row: bool = False,
):
    df = pd.read_csv(path, header=None if no_header_row else "infer", index_col=False)

    if smiles_cols is not None and rxn_cols is not None:
        smiss = df[smiles_cols].T.values.tolist()
        rxnss = df[rxn_cols].T.values.tolist()
        input_cols = [*smiles_cols, *rxn_cols]
    elif smiles_cols is not None and rxn_cols is None:
        smiss = df[smiles_cols].T.values.tolist()
        rxnss = None
        input_cols = smiles_cols
    elif smiles_cols is None and rxn_cols is not None:
        smiss = None
        rxnss = df[rxn_cols].T.values.tolist()
        input_cols = rxn_cols
    else:
        smiss = df.iloc[:, [0]].T.values.tolist()
        rxnss = None
        input_cols = [df.columns[0]]

    descriptor_cols = list(descriptor_cols or [])
    X_d_extra = df[descriptor_cols].to_numpy(np.single) if descriptor_cols else None

    if target_cols is None:
        target_cols = list(
            column
            for column in df.columns
            if column
            not in set(  # if splits or weight is None, df.columns will never have None
                input_cols + (ignore_cols or []) + descriptor_cols + [splits_col] + [weight_col]
            )
        )

    Y = df[target_cols]
    weights = None if weight_col is None else df[weight_col].to_numpy(np.single)

    if bounded:
        Y = Y.astype(str)
        lt_mask = Y.applymap(lambda x: "<" in x).to_numpy()
        gt_mask = Y.applymap(lambda x: ">" in x).to_numpy()
        Y = Y.applymap(lambda x: x.strip("<").strip(">")).to_numpy(np.single)
    else:
        Y = Y.to_numpy(np.single)
        lt_mask = None
        gt_mask = None

    return smiss, rxnss, Y, weights, lt_mask, gt_mask, X_d_extra


def get_column_names(
    path: PathLike,
    smiles_cols: Sequence[str] | None,
    rxn_cols: Sequence[str] | None,
    target_cols: Sequence[str] | None,
    ignore_cols: Sequence[str] | None,
    splits_col: str | None,
    weight_col: str | None,
    no_header_row: bool = False,
) -> tuple[list[str], list[str]]:
    df_cols = pd.read_csv(path, index_col=False, nrows=0).columns.tolist()

    if no_header_row:
        return ["SMILES"], ["pred_" + str(i) for i in range((len(df_cols) - 1))]

    input_cols = (smiles_cols or []) + (rxn_cols or [])

    if len(input_cols) == 0:
        input_cols = [df_cols[0]]

    if target_cols is None:
        target_cols = list(
            column
            for column in df_cols
            if column
            not in set(
                input_cols + (ignore_cols or []) + ([splits_col] or []) + ([weight_col] or [])
            )
        )

    return input_cols, target_cols


def make_datapoints(
    smiss: list[list[str]] | None,
    rxnss: list[list[str]] | None,
    Y: np.ndarray,
    weights: np.ndarray | None,
    lt_mask: np.ndarray | None,
    gt_mask: np.ndarray | None,
    X_d: np.ndarray | None,
    V_fss: list[list[np.ndarray] | list[None]] | None,
    E_fss: list[list[np.ndarray] | list[None]] | None,
    V_dss: list[list[np.ndarray] | list[None]] | None,
    molecule_featurizers: list[str] | None,
    keep_h: bool,
    add_h: bool,
    ignore_stereo: bool,
    reorder_atoms: bool,
    use_cuikmolmaker_featurization: bool,
) -> tuple[list[list[MoleculeDatapoint | LazyMoleculeDatapoint]], list[list[ReactionDatapoint]]]:
    """Make the :class:`MoleculeDatapoint`s and :class:`ReactionDatapoint`s for a given
    dataset.

    Parameters
    ----------
    smiss : list[list[str]] | None
        a list of ``j`` lists of ``n`` SMILES strings, where ``j`` is the number of molecules per
        datapoint and ``n`` is the number of datapoints. If ``None``, the corresponding list of
        :class:`MoleculeDatapoint`\s will be empty.
    rxnss : list[list[str]] | None
        a list of ``k`` lists of ``n`` reaction SMILES strings, where ``k`` is the number of
        reactions per datapoint. If ``None``, the corresponding list of :class:`ReactionDatapoint`\s
        will be empty.
    Y : np.ndarray
        the target values of shape ``n x m``, where ``m`` is the number of targets
    weights : np.ndarray | None
        the weights of the datapoints to use in the loss function of shape ``n``. If ``None``,
        the weights all default to 1.
    lt_mask : np.ndarray | None
        a boolean mask of shape ``n x m`` indicating whether the targets are less than inequality
        targets. If ``None``, ``lt_mask`` for all datapoints will be ``None``.
    gt_mask : np.ndarray | None
        a boolean mask of shape ``n x m`` indicating whether the targets are greater than inequality
        targets. If ``None``, ``gt_mask`` for all datapoints will be ``None``.
    X_d : np.ndarray | None
        the extra descriptors of shape ``n x p``, where ``p`` is the number of extra descriptors. If
        ``None``, ``x_d`` for all datapoints will be ``None``.
    V_fss : list[list[np.ndarray] | list[None]] | None
        a list of ``j`` lists of ``n`` np.ndarrays each of shape ``v_jn x q_j``, where ``v_jn`` is
        the number of atoms in the j-th molecule of the n-th datapoint and ``q_j`` is the number of
        extra atom features used for the j-th molecules. Any of the ``j`` lists can be a list of
        None values if the corresponding component does not use extra atom features. If ``None``,
        ``V_f`` for all datapoints will be ``None``.
    E_fss : list[list[np.ndarray] | list[None]] | None
        a list of ``j`` lists of ``n`` np.ndarrays each of shape ``e_jn x r_j``, where ``e_jn`` is
        the number of bonds in the j-th molecule of the n-th datapoint and ``r_j`` is the number of
        extra bond features used for the j-th molecules. Any of the ``j`` lists can be a list of
        None values if the corresponding component does not use extra bond features. If ``None``,
        ``E_f`` for all datapoints will be ``None``.
    V_dss : list[list[np.ndarray] | list[None]] | None
        a list of ``j`` lists of ``n`` np.ndarrays each of shape ``v_jn x s_j``, where ``s_j`` is
        the number of extra atom descriptors used for the j-th molecules. Any of the ``j`` lists can
        be a list of None values if the corresponding component does not use extra atom features. If
        ``None``, ``V_d`` for all datapoints will be ``None``.
    molecule_featurizers : list[str] | None
        a list of molecule featurizer names to generate additional molecule features to use as extra
        descriptors. If there are multiple molecules per datapoint, the featurizers will be applied
        to each molecule and concatenated. Note that a :code:`ReactionDatapoint` has two
        RDKit :class:`~rdkit.Chem.Mol` objects, reactant(s) and product(s). Each
        ``molecule_featurizer`` will be applied to both of these objects.
    keep_h : bool
        whether to keep hydrogen atoms
    add_h : bool
        whether to add hydrogen atoms
    ignore_stereo : bool
        whether to ignore stereo information

    Returns
    -------
    list[list[MoleculeDatapoint]]
        a list of ``j`` lists of ``n`` :class:`MoleculeDatapoint`\s
    list[list[ReactionDatapoint]]
        a list of ``k`` lists of ``n`` :class:`ReactionDatapoint`\s
    .. note::
        either ``j`` or ``k`` may be 0, in which case the corresponding list will be empty.

    Raises
    ------
    ValueError
        if both ``smiss`` and ``rxnss`` are ``None``.
        if ``smiss`` and ``rxnss`` are both given and have different lengths.
    """
    if smiss is None and rxnss is None:
        raise ValueError("args 'smiss' and 'rnxss' were both `None`!")
    elif rxnss is None:
        N = len(smiss[0])
        rxnss = []
    elif smiss is None:
        N = len(rxnss[0])
        smiss = []
    elif len(smiss[0]) != len(rxnss[0]):
        raise ValueError(
            f"args 'smiss' and 'rxnss' must have same length! got {len(smiss[0])} and {len(rxnss[0])}"
        )
    else:
        N = len(smiss[0])

    weights = np.ones(N, dtype=np.single) if weights is None else weights
    gt_mask = [None] * N if gt_mask is None else gt_mask
    lt_mask = [None] * N if lt_mask is None else lt_mask
    n_mols = len(smiss) if smiss else 0
    V_fss = [[None] * N] * n_mols if V_fss is None else V_fss
    E_fss = [[None] * N] * n_mols if E_fss is None else E_fss
    V_dss = [[None] * N] * n_mols if V_dss is None else V_dss
    # if X_d is None and molecule_featurizers is None:
    #     X_d = [None] * N

    if use_cuikmolmaker_featurization:
        # Form `LazyMoleculeDatapoint`s first and then compute and add molecule features
        mol_data = [
            [
                LazyMoleculeDatapoint(
                    smiles=smiss[smi_idx][i],
                    _keep_h=keep_h,
                    _add_h=add_h,
                    _ignore_stereo=ignore_stereo,
                    _reorder_atoms=reorder_atoms,
                    name=smis[i],
                    y=Y[i],
                    weight=weights[i],
                    gt_mask=gt_mask[i],
                    lt_mask=lt_mask[i],
                    # x_d=X_d[i],
                    x_phase=None,
                    V_f=V_fss[smi_idx][i],
                    E_f=E_fss[smi_idx][i],
                    V_d=V_dss[smi_idx][i],
                )
                for i in range(N)
            ]
            for smi_idx, smis in enumerate(smiss)
        ]
        if X_d is None and molecule_featurizers is None:
            X_d = [None] * N
        elif molecule_featurizers is None:
            pass
        else:
            if len(smiss) > 0:
                molecule_featurizers_fns = [
                    MoleculeFeaturizerRegistry[mf]() for mf in molecule_featurizers
                ]

                if len(smiss) > 0:
                    mol_descriptors = np.hstack(
                        [
                            np.vstack(
                                [
                                    np.hstack([mf(mol_dp.mol) for mf in molecule_featurizers_fns])
                                    for mol_dp in mol_dp_list
                                ]
                            )
                            for mol_dp_list in mol_data
                        ]
                    )
                    if X_d is None:
                        X_d = mol_descriptors
                    else:
                        X_d = np.hstack([X_d, mol_descriptors])

                [
                    setattr(mol_data[mol_idx][i], "x_d", X_d[i])
                    for mol_idx, _ in enumerate(smiss)
                    for i in range(N)
                ]
    else:
        # Compute molecule features first and then form `MoleculeDatapoint`s
        if len(smiss) > 0:
            molss = [
                [make_mol(smi, keep_h, add_h, ignore_stereo, reorder_atoms) for smi in smis]
                for smis in smiss
            ]
        if X_d is None and molecule_featurizers is None:
            X_d = [None] * N
        elif molecule_featurizers is None:
            pass
        else:
            molecule_featurizers_fns = [
                MoleculeFeaturizerRegistry[mf]() for mf in molecule_featurizers
            ]

            if len(smiss) > 0:
                mol_descriptors = np.hstack(
                    [
                        np.vstack(
                            [
                                np.hstack([mf(mol) for mf in molecule_featurizers_fns])
                                for mol in mols
                            ]
                        )
                        for mols in molss
                    ]
                )
                if X_d is None:
                    X_d = mol_descriptors
                else:
                    X_d = np.hstack([X_d, mol_descriptors])
        mol_data = [
            [
                MoleculeDatapoint(
                    mol=molss[mol_idx][i],
                    name=smis[i],
                    y=Y[i],
                    weight=weights[i],
                    gt_mask=gt_mask[i],
                    lt_mask=lt_mask[i],
                    x_d=X_d[i],
                    x_phase=None,
                    V_f=V_fss[mol_idx][i],
                    E_f=E_fss[mol_idx][i],
                    V_d=V_dss[mol_idx][i],
                )
                for i in range(N)
            ]
            for mol_idx, smis in enumerate(smiss)
        ]

    if len(rxnss) > 0:
        rctss = [
            [
                make_mol(
                    f"{rct_smi}.{agt_smi}" if agt_smi else rct_smi,
                    keep_h,
                    add_h,
                    ignore_stereo,
                    reorder_atoms,
                )
                for rct_smi, agt_smi, _ in (rxn.split(">") for rxn in rxns)
            ]
            for rxns in rxnss
        ]
        pdtss = [
            [
                make_mol(pdt_smi, keep_h, add_h, ignore_stereo, reorder_atoms)
                for _, _, pdt_smi in (rxn.split(">") for rxn in rxns)
            ]
            for rxns in rxnss
        ]

        if molecule_featurizers is not None:
            molecule_featurizers_fns = [
                MoleculeFeaturizerRegistry[mf]() for mf in molecule_featurizers
            ]
            if len(rxnss) > 0:
                rct_pdt_descriptors = np.hstack(
                    [
                        np.vstack(
                            [
                                np.hstack(
                                    [
                                        mf(mol)
                                        for mf in molecule_featurizers_fns
                                        for mol in (rct, pdt)
                                    ]
                                )
                                for rct, pdt in zip(rcts, pdts)
                            ]
                        )
                        for rcts, pdts in zip(rctss, pdtss)
                    ]
                )
                if X_d is None:
                    X_d = rct_pdt_descriptors
                else:
                    X_d = np.hstack([X_d, rct_pdt_descriptors])

    rxn_data = [
        [
            ReactionDatapoint(
                rct=rctss[rxn_idx][i],
                pdt=pdtss[rxn_idx][i],
                name=rxns[i],
                y=Y[i],
                weight=weights[i],
                gt_mask=gt_mask[i],
                lt_mask=lt_mask[i],
                x_d=X_d[i],
                x_phase=None,
            )
            for i in range(N)
        ]
        for rxn_idx, rxns in enumerate(rxnss)
    ]

    return mol_data, rxn_data


def build_data_from_files(
    p_data: PathLike,
    no_header_row: bool,
    smiles_cols: Sequence[str] | None,
    rxn_cols: Sequence[str] | None,
    target_cols: Sequence[str] | None,
    ignore_cols: Sequence[str] | None,
    splits_col: str | None,
    weight_col: str | None,
    bounded: bool,
    p_descriptors: PathLike,
    p_atom_feats: dict[int, PathLike],
    p_bond_feats: dict[int, PathLike],
    p_atom_descs: dict[int, PathLike],
    descriptor_cols: Sequence[str] | None = None,
    **featurization_kwargs: Mapping,
) -> list[list[MoleculeDatapoint] | list[ReactionDatapoint]]:
    smiss, rxnss, Y, weights, lt_mask, gt_mask, X_d_extra = parse_csv(
        p_data,
        smiles_cols,
        rxn_cols,
        target_cols,
        ignore_cols,
        splits_col,
        weight_col,
        descriptor_cols,
        bounded,
        no_header_row,
    )
    n_molecules = len(smiss) if smiss is not None else 0
    n_datapoints = len(Y)

    X_ds = load_input_feats_and_descs(p_descriptors, None, None, feat_desc="X_d")
    if X_d_extra is not None:
        if X_ds is None:
            X_ds = X_d_extra
        else:
            X_ds = np.hstack([X_ds, X_d_extra])

    V_fss = load_input_feats_and_descs(p_atom_feats, n_molecules, n_datapoints, feat_desc="V_f")
    E_fss = load_input_feats_and_descs(p_bond_feats, n_molecules, n_datapoints, feat_desc="E_f")
    V_dss = load_input_feats_and_descs(p_atom_descs, n_molecules, n_datapoints, feat_desc="V_d")

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


def load_input_feats_and_descs(
    paths: dict[int, PathLike] | PathLike,
    n_molecules: int | None,
    n_datapoints: int | None,
    feat_desc: str,
):
    if paths is None:
        return None

    match feat_desc:
        case "X_d":
            path = paths
            loaded_feature = np.load(path)
            features = loaded_feature["arr_0"]

        case _:
            for index in paths:
                if index >= n_molecules:
                    raise ValueError(
                        f"For {n_molecules} molecules, atom/bond features/descriptors can only be "
                        f"specified for indices 0-{n_molecules - 1}! Got index {index}."
                    )

            features = []
            for idx in range(n_molecules):
                path = paths.get(idx, None)

                if path is not None:
                    loaded_feature = np.load(path)
                    loaded_feature = [
                        loaded_feature[f"arr_{i}"] for i in range(len(loaded_feature))
                    ]
                else:
                    loaded_feature = [None] * n_datapoints

                features.append(loaded_feature)
    return features


def make_dataset(
    data: Sequence[MoleculeDatapoint]
    | Sequence[MolAtomBondDatapoint]
    | Sequence[ReactionDatapoint],
    reaction_mode: Literal[*tuple(RxnMode.keys())] = "REAC_DIFF",
    multi_hot_atom_featurizer_mode: Literal["V1", "V2", "ORGANIC", "RIGR"] = "V2",
    cuikmolmaker_featurization: bool = False,
) -> MoleculeDataset | CuikmolmakerDataset | MolAtomBondDataset | ReactionDataset:
    atom_featurizer = get_multi_hot_atom_featurizer(multi_hot_atom_featurizer_mode)
    match multi_hot_atom_featurizer_mode:
        case "RIGR":
            bond_featurizer = RIGRBondFeaturizer()
        case "V1" | "V2" | "ORGANIC":
            bond_featurizer = MultiHotBondFeaturizer()
        case _:
            raise TypeError(
                f"Unsupported atom featurizer mode '{multi_hot_atom_featurizer_mode=}'!"
            )

    if isinstance(data[0], MolAtomBondDatapoint):
        extra_atom_fdim = data[0].V_f.shape[1] if data[0].V_f is not None else 0
        extra_bond_fdim = data[0].E_f.shape[1] if data[0].E_f is not None else 0
        featurizer = SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            extra_atom_fdim=extra_atom_fdim,
            extra_bond_fdim=extra_bond_fdim,
        )
        return MolAtomBondDataset(data, featurizer)

    if isinstance(data[0], MoleculeDatapoint) or isinstance(data[0], LazyMoleculeDatapoint):
        if cuikmolmaker_featurization:
            add_h = data[0]._add_h
            featurizer = CuikmolmakerMolGraphFeaturizer(
                atom_featurizer=atom_featurizer,
                bond_featurizer=bond_featurizer,
                atom_featurizer_mode=multi_hot_atom_featurizer_mode,
                add_h=add_h,
            )
            return CuikmolmakerDataset(data, featurizer)

        extra_atom_fdim = data[0].V_f.shape[1] if data[0].V_f is not None else 0
        extra_bond_fdim = data[0].E_f.shape[1] if data[0].E_f is not None else 0
        featurizer = SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            extra_atom_fdim=extra_atom_fdim,
            extra_bond_fdim=extra_bond_fdim,
        )
        return MoleculeDataset(data, featurizer)

    featurizer = CondensedGraphOfReactionFeaturizer(
        mode_=reaction_mode, atom_featurizer=atom_featurizer, bond_featurizer=bond_featurizer
    )

    return ReactionDataset(data, featurizer)


def parse_indices(idxs):
    """Parses a string of indices into a list of integers. e.g. '0,1,2-4' -> [0, 1, 2, 3, 4]"""
    if isinstance(idxs, str):
        indices = []
        for idx in idxs.split(","):
            if "-" in idx:
                start, end = map(int, idx.split("-"))
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(idx))
        return indices
    return idxs


def parse_activation(cls: type[nn.Module], arguments: list | None) -> nn.Module:
    """Parse arguments and instantiate an activation function"""
    posargs, kwargs = [], {}
    if arguments is not None:
        for item in arguments:
            if isinstance(item, dict):
                kwargs.update(item)
            else:
                posargs.append(item)
    return cls(*posargs, **kwargs)
