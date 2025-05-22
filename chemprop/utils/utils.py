from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Iterator

import numpy as np
from rdkit import Chem


class EnumMapping(StrEnum):
    @classmethod
    def get(cls, name: str | EnumMapping) -> EnumMapping:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise KeyError(
                f"Unsupported {cls.__name__} member! got: '{name}'. expected one of: {', '.join(cls.keys())}"
            )

    @classmethod
    def keys(cls) -> Iterator[str]:
        return (e.name for e in cls)

    @classmethod
    def values(cls) -> Iterator[str]:
        return (e.value for e in cls)

    @classmethod
    def items(cls) -> Iterator[tuple[str, str]]:
        return zip(cls.keys(), cls.values())


def make_mol(
    smi: str,
    keep_h: bool = False,
    add_h: bool = False,
    ignore_stereo: bool = False,
    reorder_atoms: bool = False,
) -> Chem.Mol:
    """build an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool, optional
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps
        them if they are specified. Default is False.
    add_h : bool, optional
        whether to add hydrogens to the molecule. Default is False.
    ignore_stereo : bool, optional
        whether to ignore stereochemical information (R/S and Cis/Trans) when constructing the molecule. Default is False.
    reorder_atoms : bool, optional
        whether to reorder the atoms in the molecule by their atom map numbers. This is useful when
        the order of atoms in the SMILES string does not match the atom mapping, e.g. '[F:2][Cl:1]'.
        Default is False. NOTE: This does not reorder the bonds.

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h
    mol = Chem.MolFromSmiles(smi, params)

    if mol is None:
        raise RuntimeError(f"SMILES {smi} is invalid! (RDKit returned None)")

    if add_h:
        mol = Chem.AddHs(mol)

    if ignore_stereo:
        for atom in mol.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in mol.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)

    if reorder_atoms:
        atom_map_numbers = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
        new_order = np.argsort(atom_map_numbers).tolist()
        mol = Chem.rdmolops.RenumberAtoms(mol, new_order)

    return mol


def pretty_shape(shape: Iterable[int]) -> str:
    """Make a pretty string from an input shape

    Example
    --------
    >>> X = np.random.rand(10, 4)
    >>> X.shape
    (10, 4)
    >>> pretty_shape(X.shape)
    '10 x 4'
    """
    return " x ".join(map(str, shape))
