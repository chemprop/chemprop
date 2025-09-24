import ast
from os import PathLike
from typing import Sequence

import numpy as np
import pandas as pd

from chemprop.data import MolAtomBondDatapoint
from chemprop.featurizers.molecule import MoleculeFeaturizerRegistry
from chemprop.utils import create_and_call_object, make_mol, parallel_execute


def build_MAB_data_from_files(
    p_data: PathLike,
    smiles_cols: Sequence[str] | None,
    mol_target_cols: Sequence[str] | None,
    atom_target_cols: Sequence[str] | None,
    bond_target_cols: Sequence[str] | None,
    weight_col: str | None,
    bounded: bool,
    p_descriptors: PathLike | None,
    p_atom_feats: dict[int, PathLike] | None,
    p_bond_feats: dict[int, PathLike] | None,
    p_atom_descs: dict[int, PathLike] | None,
    p_bond_descs: dict[int, PathLike] | None,
    p_constraints: PathLike | None,
    constraints_cols_to_target_cols: dict[str, int] | None,
    molecule_featurizers: Sequence[str] | None,
    descriptor_cols: Sequence[str] | None = None,
    n_workers: int = 0,
    **make_mol_kwargs,
):
    df = pd.read_csv(p_data, index_col=False)

    X_d_extra = df[descriptor_cols] if descriptor_cols else None

    smis = df[smiles_cols[0]].values if smiles_cols is not None else df.iloc[:, 0].values
    mols = parallel_execute(
        make_mol, [(smi,) + tuple(make_mol_kwargs.values()) for smi in smis], n_workers=n_workers
    )
    n_datapoints = len(mols)

    weights = (
        df[weight_col].values if weight_col is not None else np.ones(n_datapoints, dtype=np.single)
    )

    X_ds = np.load(p_descriptors)["arr_0"] if p_descriptors else [None] * n_datapoints
    if X_d_extra is not None:
        if X_ds[0] is None:
            X_ds = X_d_extra
        else:
            X_ds = np.hstack([X_ds, X_d_extra])

    loaded_arrays = np.load(p_atom_feats[0]) if p_atom_feats else None
    V_fs = (
        [loaded_arrays[f"arr_{i}"] for i in range(n_datapoints)]
        if loaded_arrays
        else [None] * n_datapoints
    )
    loaded_arrays = np.load(p_atom_descs[0]) if p_atom_descs else None
    V_ds = (
        [loaded_arrays[f"arr_{i}"] for i in range(n_datapoints)]
        if loaded_arrays
        else [None] * n_datapoints
    )
    loaded_arrays = np.load(p_bond_feats[0]) if p_bond_feats else None
    E_fs = (
        [loaded_arrays[f"arr_{i}"] for i in range(n_datapoints)]
        if loaded_arrays
        else [None] * n_datapoints
    )
    loaded_arrays = np.load(p_bond_descs[0]) if p_bond_descs else None
    E_ds = (
        [np.repeat(loaded_arrays[f"arr_{i}"], repeats=2, axis=0) for i in range(n_datapoints)]
        if loaded_arrays
        else [None] * n_datapoints
    )

    if molecule_featurizers is not None:
        molecule_featurizers = [MoleculeFeaturizerRegistry[mf]() for mf in molecule_featurizers]
        mol_descriptors = np.hstack(
            [
                np.vstack(
                    parallel_execute(
                        create_and_call_object,
                        [(mf.__class__, (mol,)) for mol in mols],
                        n_workers=n_workers,
                    )
                )
                for mf in molecule_featurizers
            ]
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
        mol_ys = df[mol_target_cols] if mol_target_cols is not None else None
        atoms_ys = df[atom_target_cols] if atom_target_cols is not None else None
        bonds_ys = df[bond_target_cols] if bond_target_cols is not None else None

        if mol_ys is not None:
            mol_ys = mol_ys.astype(str)
            lt_mask = mol_ys.map(lambda x: "<" in x).to_numpy()
            gt_mask = mol_ys.map(lambda x: ">" in x).to_numpy()
            mol_ys = mol_ys.map(lambda x: x.strip("<").strip(">")).to_numpy(np.single)
        else:
            mol_ys = [None] * n_datapoints

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
        else:
            atoms_ys = [None] * n_datapoints

        if bonds_ys is not None:
            bonds_ys = bonds_ys.map(ast.literal_eval)
            # Doesn't yet work with giving bonds_ys as a list of lists
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
            bonds_ys = [None] * n_datapoints

    else:
        mol_ys = (
            df[mol_target_cols].values if mol_target_cols is not None else [None] * n_datapoints
        )
        atoms_ys = df[atom_target_cols].values if atom_target_cols is not None else None
        bonds_ys = df[bond_target_cols].values if bond_target_cols is not None else None

        if atoms_ys is not None:
            atoms_ys = [
                np.array([ast.literal_eval(atom_y) for atom_y in atom_ys], dtype=float).T
                for atom_ys in atoms_ys
            ]
        else:
            atoms_ys = [None] * n_datapoints
        if bonds_ys is not None:
            bonds_ys = [
                np.array([ast.literal_eval(bond_y) for bond_y in bond_ys], dtype=float).T
                for bond_ys in bonds_ys
            ]
            if bonds_ys[0].ndim == 3:
                bonds_ys_1D = []
                for mol, bonds_y in zip(mols, bonds_ys):
                    bond_vals = [0] * mol.GetNumBonds()
                    for i_bond, bond in enumerate(mol.GetBonds()):
                        bond_vals[i_bond] = bonds_y[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), :]
                    bonds_ys_1D.append(np.array(bond_vals, dtype=float))
                bonds_ys = bonds_ys_1D

        else:
            bonds_ys = [None] * n_datapoints

    atom_constraints = [None] * n_datapoints
    bond_constraints = [None] * n_datapoints

    if p_constraints:
        df_constraints = pd.read_csv(p_constraints, index_col=False)
        n_mols = len(df_constraints)

        if constraints_cols_to_target_cols is None:
            raise ValueError(
                "If using constraints, you must indicate which constraint column corresponds to "
                "which atom or bond target column. This is done by passing the names of the atom "
                "or bond target columns that correspond to each constraint column to the "
                "`--constraints-to-targets` flag"
            )

        atom_constraint_cols = [
            constraints_cols_to_target_cols.get(col) for col in atom_target_cols
        ]
        if atom_constraint_cols:
            atom_constraints = np.hstack(
                [
                    df_constraints.iloc[:, col].values.reshape(-1, 1)
                    if col is not None
                    else np.full((n_mols, 1), np.nan)
                    for col in atom_constraint_cols
                ]
            )

        bond_constraint_cols = [
            constraints_cols_to_target_cols.get(col) for col in bond_target_cols
        ]
        if bond_constraint_cols:
            bond_constraints = np.hstack(
                [
                    df_constraints.iloc[:, col].values.reshape(-1, 1)
                    if col is not None
                    else np.full((n_mols, 1), np.nan)
                    for col in bond_constraint_cols
                ]
            )

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
            atom_constraint=atom_constraints[i],
            bond_constraint=bond_constraints[i],
        )
        for i in range(n_datapoints)
    ]

    return [datapoints]
