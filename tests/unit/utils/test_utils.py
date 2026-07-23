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


def test_reorder_atoms_no_atom_map_large():
    # diphenhydramine, 19 heavy atoms, unmapped: large enough to expose an unstable sort
    smi = "CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1"
    mol = make_mol(smi, reorder_atoms=False)
    reordered_mol = make_mol(smi, reorder_atoms=True)
    assert [a.GetSymbol() for a in mol.GetAtoms()] == [
        a.GetSymbol() for a in reordered_mol.GetAtoms()
    ]


def test_reorder_atoms_partial_map():
    # unmapped atoms (map number 0) must keep their relative order, ahead of mapped atoms
    mol = make_mol("S[C:5]NO", reorder_atoms=True)
    assert [a.GetSymbol() for a in mol.GetAtoms()] == ["S", "N", "O", "C"]


def test_reorder_atoms_add_h():
    # hydrogens added by AddHs are all unmapped (map number 0) and must keep their relative order
    # every hydrogen can be distinguished by its neighbor's map number after reordering
    smi = "[cH:1]1[cH:2][cH:3][c:4]2[cH:5][cH:6][cH:7][cH:8][c:9]2[cH:10]1"
    mol = make_mol(smi, add_h=True, reorder_atoms=False)
    reordered_mol = make_mol(smi, add_h=True, reorder_atoms=True)

    def h_neighbor_maps(mol):
        return [
            a.GetNeighbors()[0].GetAtomMapNum()
            for a in mol.GetAtoms()
            if a.GetSymbol() == "H"
        ]

    assert h_neighbor_maps(mol) == h_neighbor_maps(reordered_mol)


def test_make_mol_invalid_smiles():
    with pytest.raises(RuntimeError):
        make_mol("chemprop")
