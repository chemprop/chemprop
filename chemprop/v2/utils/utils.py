from __future__ import annotations

from enum import Enum
from typing import Union

from rdkit import Chem


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    def __str__(self) -> str:
        return self.value

    @classmethod
    def get(cls, name: Union[str, AutoName]) -> AutoName:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.lower()]
        except KeyError:
            raise ValueError(f"Unsupported alias! got: '{name}'. expected one of: {cls.keys()}")

    @classmethod
    def keys(cls) -> list[str]:
        return [e.value for e in cls]


def make_mol(smi: str, keep_h: bool, add_h: bool) -> Chem.Mol:
    """build an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified
    add_h : bool
        whether to add hydrogens to the molecule

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        )
    else:
        mol = Chem.MolFromSmiles(smi)

    return Chem.AddHs(mol) if add_h else mol
