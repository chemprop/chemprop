from rdkit.Chem import Mol

Rxn = tuple[Mol, Mol]
Polymer = tuple[Mol, list[str], list[str]]
