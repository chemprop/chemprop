import ast
from os import PathLike
from typing import Sequence

import numpy as np
import pandas as pd

from chemprop.data import MolAtomBondDatapoint
from chemprop.featurizers.molecule import MoleculeFeaturizerRegistry
from chemprop.utils import make_mol


def build_MAB_data_from_files(
    p_data: PathLike,
    smiles_cols: Sequence[str],
    target_cols: Sequence[str] | None,
    atom_target_cols: Sequence[str] | None,
    bond_target_cols: Sequence[str] | None,
    weight_col: str | None,
    bounded: bool,
    p_descriptors: PathLike | None,
    p_atom_feats: dict[int, PathLike] | None,
    p_bond_feats: dict[int, PathLike] | None,
    p_atom_descs: dict[int, PathLike] | None,
    p_bond_descs: dict[int, PathLike] | None,
    molecule_featurizers: Sequence[str] | None,
    **make_mol_kwargs,
):
    df = pd.read_csv(p_data, index_col=False)
    smis = df[smiles_cols[0]].values
    mols = [make_mol(smi, **make_mol_kwargs) for smi in smis]
    weights = df[weight_col].values if weight_col is not None else None

    n_datapoints = len(mols)

    X_ds = np.load(p_descriptors)["arr_0"] if p_descriptors else [None] * n_datapoints
    V_fs = (
        [np.load(p_atom_feats[0])[f"arr_{i}"] for i in range(n_datapoints)]
        if p_atom_feats
        else [None] * n_datapoints
    )
    V_ds = (
        [np.load(p_atom_descs[0])[f"arr_{i}"] for i in range(n_datapoints)]
        if p_atom_descs
        else [None] * n_datapoints
    )
    E_fs = (
        [np.load(p_bond_feats[0])[f"arr_{i}"] for i in range(n_datapoints)]
        if p_bond_feats
        else [None] * n_datapoints
    )
    E_ds = (
        [np.load(p_bond_descs[0])[f"arr_{i}"] for i in range(n_datapoints)]
        if p_bond_descs
        else [None] * n_datapoints
    )

    if molecule_featurizers is not None:
        molecule_featurizers = [MoleculeFeaturizerRegistry[mf]() for mf in molecule_featurizers]
        mol_descriptors = np.vstack(
            [np.hstack([mf(mol) for mf in molecule_featurizers]) for mol in mols]
        )
        if X_ds[0] is not None:
            X_ds = np.hstack([X_ds, mol_descriptors])

    lt_mask = [None] * n_datapoints
    gt_mask = [None] * n_datapoints
    atom_lt_masks = [None] * n_datapoints
    atom_gt_masks = [None] * n_datapoints
    bond_lt_masks = [None] * n_datapoints
    bond_gt_masks = [None] * n_datapoints

    if bounded:
        mol_ys = df[target_cols] if target_cols is not None else None
        atoms_ys = df[atom_target_cols] if atom_target_cols is not None else None
        bonds_ys = df[bond_target_cols] if bond_target_cols is not None else None

        if mol_ys is not None:
            mol_ys = mol_ys.astype(str)
            lt_mask = mol_ys.map(lambda x: "<" in x).to_numpy()
            gt_mask = mol_ys.map(lambda x: ">" in x).to_numpy()
            mol_ys = mol_ys.map(lambda x: x.strip("<").strip(">")).to_numpy(np.single)

        if atoms_ys is not None:
            atoms_ys = atoms_ys.map(ast.literal_eval)
            atom_lt_masks = atoms_ys.map(lambda L: ["<" in v if v else False for v in L])
            atom_gt_masks = atoms_ys.map(lambda L: [">" in v if v else False for v in L])

            atom_lt_masks = atom_lt_masks.apply(
                lambda row: np.vstack(row.values).T, axis=1
            ).tolist()
            atom_gt_masks = atom_gt_masks.apply(
                lambda row: np.vstack(row.values).T, axis=1
            ).tolist()
            atoms_ys = atoms_ys.map(
                lambda L: np.array(
                    [v.strip("<").strip(">") if v else "nan" for v in L], dtype=np.single
                )
            )
            atoms_ys = atoms_ys.apply(lambda row: np.vstack(row.values).T, axis=1).tolist()

        if bonds_ys is not None:
            bonds_ys = bonds_ys.map(ast.literal_eval)
            bond_lt_masks = bonds_ys.map(lambda L: ["<" in v if v else False for v in L])
            bond_gt_masks = bonds_ys.map(lambda L: [">" in v if v else False for v in L])

            bond_lt_masks = bond_lt_masks.apply(
                lambda row: np.vstack(row.values).T, axis=1
            ).tolist()
            bond_gt_masks = bond_gt_masks.apply(
                lambda row: np.vstack(row.values).T, axis=1
            ).tolist()

            bond_lt_masks = [bond_lt_mask.astype(bool) for bond_lt_mask in bond_lt_masks]
            bond_gt_masks = [bond_gt_mask.astype(bool) for bond_gt_mask in bond_gt_masks]

            bonds_ys = bonds_ys.map(
                lambda L: np.array(
                    [v.strip("<").strip(">") if v else "nan" for v in L], dtype=np.single
                )
            )
            bonds_ys = bonds_ys.apply(lambda row: np.vstack(row.values).T, axis=1).tolist()

    else:
        mol_ys = df[target_cols].values if target_cols is not None else None
        atoms_ys = df[atom_target_cols].values if atom_target_cols is not None else None
        bonds_ys = df[bond_target_cols].values if bond_target_cols is not None else None

        if atoms_ys is not None:
            atoms_ys = [
                np.array([ast.literal_eval(atom_y) for atom_y in atom_ys], dtype=float).T
                for atom_ys in atoms_ys
            ]
        if bonds_ys is not None:
            bonds_ys = [
                np.array([ast.literal_eval(bond_y) for bond_y in bond_ys], dtype=float).T
                for bond_ys in bonds_ys
            ]

    datapoints = [
        MolAtomBondDatapoint(
            mol=mols[i],
            name=smis[i],
            y=mol_ys[i],
            atom_y=atoms_ys[i],
            bond_y=bonds_ys[i],
            weight=weights[i],
            x_d=X_ds[i],
            V_f=V_fs[i],
            V_d=V_ds[i],
            E_f=E_fs[i],
            E_d=E_ds[i],
            lt_mask=lt_mask[i],
            gt_mask=gt_mask[i],
            atom_lt_mask=atom_lt_masks[i],
            atom_gt_mask=atom_gt_masks[i],
            bond_lt_mask=bond_lt_masks[i],
            bond_gt_mask=bond_gt_masks[i],
        )
        for i in range(n_datapoints)
    ]

    return [datapoints]
