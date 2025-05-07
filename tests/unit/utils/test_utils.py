import pytest

from chemprop.utils import make_mol


def test_no_keep_h():
    mol = make_mol("[H]C", keep_h=False)
    assert mol.GetNumAtoms() == 1


def test_keep_h():
    mol = make_mol("[H]C", keep_h=True)
    assert mol.GetNumAtoms() == 2


def test_add_h():
    mol = make_mol("[H]C", add_h=True)
    assert mol.GetNumAtoms() == 5


def test_no_reorder_atoms():
    mol = make_mol("[CH3:2][OH:1]", reorder_atoms=False)
    assert mol.GetAtomWithIdx(0).GetSymbol() == "C"


def test_reorder_atoms():
    mol = make_mol("[CH3:2][OH:1]", reorder_atoms=True)
    assert mol.GetAtomWithIdx(0).GetSymbol() == "O"


def test_reorder_atoms_no_atom_map():
    mol = make_mol("CCO", reorder_atoms=False)
    reordered_mol = make_mol("CCO", reorder_atoms=True)
    assert all(
        [
            mol.GetAtomWithIdx(i).GetSymbol() == reordered_mol.GetAtomWithIdx(i).GetSymbol()
            for i in range(mol.GetNumAtoms())
        ]
    )


def test_make_mol_invalid_smiles():
    with pytest.raises(RuntimeError):
        make_mol("chemprop")
