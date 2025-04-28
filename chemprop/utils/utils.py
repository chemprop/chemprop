from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Iterator

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
                f"Unsupported {cls.__name__} member! got: '{name}'. expected one of: {cls.keys()}"
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


def make_mol(smi: str, keep_h: bool, add_h: bool, ignore_chirality: bool = False) -> Chem.Mol:
    """build an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified
    add_h : bool
        If True, adds hydrogens to the molecule.
    ignore_chirality : bool, optional
        If True, ignores chirality information when constructing the molecule. Default is False.

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

    if mol is None:
        raise RuntimeError(f"SMILES {smi} is invalid! (RDKit returned None)")

    if add_h:
        mol = Chem.AddHs(mol)

    if ignore_chirality:
        for atom in mol.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    return mol


def make_polymer_mol(smi: str, keep_h: bool, add_h: bool, fragment_weights: list, ignore_chirality: bool = False) -> Chem.Mol:
    """
    Builds an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified
    add_h : bool
        whether to add hydrogens to the molecule
    fragment_weights: list
        list of monomer fractions for each fragment in smiles. Only used when the input is a polymer.
    ignore_chirality : bool, optional
        If True, ignores chirality information when constructing the molecule. Default is False.

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    # Check the input is correct. We need the same number of fragments and their weights.
    num_frags = len(smi.split("."))
    if len(fragment_weights) != num_frags:
        raise ValueError(f'The number of input monomers/fragments ({num_frags}) does not match the number of input weights ({len(fragment_weights)})')
    # Ensure all fragment weights are floats
    fragment_weights = [float(x) for x in fragment_weights]
    # If it all looks good, we create one molecule object per fragment and combine the fragments into
    # a single molecule object
    mols = []
    for s, w in zip(smi.split('.'), fragment_weights):
        m = make_mol(s, keep_h, add_h, ignore_chirality=ignore_chirality)
        mols.append(m)
    # Combine all the mols into a single mol object
    mol = mols.pop(0)
    while len(mols) > 0:
        m2 = mols.pop(0)
        mol = Chem.CombineMols(mol, m2)

    return mol


def remove_wildcard_atoms(rwmol):
    """
    removes wildcard atoms from an RDKit Mol
    """
    indicies = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
    while len(indicies) > 0:
        rwmol.RemoveAtom(indicies[0])
        indicies = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
    Chem.SanitizeMol(rwmol, Chem.SanitizeFlags.SANITIZE_ALL)

    return rwmol


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
