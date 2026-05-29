import pytest
from rdkit.Chem import ChiralType

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


# CXSMILES wD/wU tests


def test_cxsmiles_wd_sets_chiral_tags():
    """wD (wedge down) sets CW/CCW chiral tags on atropisomer axis atoms."""
    cxsmiles = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl |wD:14.16|"
    mol = make_mol(cxsmiles)

    assert mol.GetAtomWithIdx(14).GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CW
    assert mol.GetAtomWithIdx(16).GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CCW


def test_cxsmiles_wu_leaves_unspecified():
    """wU (wedge unspecified) leaves atropisomer axis atoms as CHI_UNSPECIFIED."""
    cxsmiles = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl |wU:14.16|"
    mol = make_mol(cxsmiles)

    assert mol.GetAtomWithIdx(14).GetChiralTag() == ChiralType.CHI_UNSPECIFIED
    assert mol.GetAtomWithIdx(16).GetChiralTag() == ChiralType.CHI_UNSPECIFIED


def test_cxsmiles_wd_produces_different_features_than_wu():
    """wD and wU on the same bond produce different molecular representations."""
    import numpy as np

    from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer

    cxsmiles_wd = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl |wD:14.16|"
    cxsmiles_wu = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl |wU:14.16|"

    mol_wd = make_mol(cxsmiles_wd)
    mol_wu = make_mol(cxsmiles_wu)

    mg_featurizer = SimpleMoleculeMolGraphFeaturizer()
    graph_wd = mg_featurizer(mol_wd)
    graph_wu = mg_featurizer(mol_wu)

    sum_wd = graph_wd.V.sum(axis=0)
    sum_wu = graph_wu.V.sum(axis=0)
    assert not np.array_equal(sum_wd, sum_wu)


def test_cxsmiles_preserves_existing_chiral_tags():
    """CXSMILES wD does not overwrite existing chiral tags from SMILES."""
    # C[C@@H](O) has a tetrahedral chiral center at atom 1
    cxsmiles = "C[C@@H](O)CC1=CC=C(C)C=C1 |wD:1.2|"
    mol = make_mol(cxsmiles)

    # Atom 1 should keep its original chiral tag from [C@@H]
    assert mol.GetAtomWithIdx(1).GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CW
    # Atom 2 should get the CCW tag from wD
    assert mol.GetAtomWithIdx(2).GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CCW


def test_cxsmiles_multiple_constraints():
    """Multiple wD/wU constraints on the same SMILES are all processed."""
    # wD:14.15 sets tags on atoms 14,15; wU:7.7 leaves atoms 7,7 unspecified
    cxsmiles = "O=C1C=CNC(=O)N1C(=C)[C@]([C@H](C)Br)([C@@H](F)C)[C@H](Cl)C |wD:14.15,wU:7.7|"
    mol = make_mol(cxsmiles)

    # wD:14.15 should set CW on atom 14, CCW on atom 15
    assert mol.GetAtomWithIdx(14).GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CW
    assert mol.GetAtomWithIdx(15).GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CCW


def test_cxsmiles_invalid_atom_indices_ignored():
    """CXSMILES with atom indices beyond the molecule size are silently ignored."""
    cxsmiles = "CCO |wD:99.100|"
    mol = make_mol(cxsmiles)

    # Should parse successfully, no chiral tags set (atoms 99,100 don't exist)
    assert mol.GetNumAtoms() == 3
    for atom in mol.GetAtoms():
        assert atom.GetChiralTag() == ChiralType.CHI_UNSPECIFIED


def test_cxsmiles_no_metadata_unchanged():
    """SMILES without CXSMILES metadata are unaffected."""
    smi = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl"
    mol = make_mol(smi)

    for atom in mol.GetAtoms():
        assert atom.GetChiralTag() == ChiralType.CHI_UNSPECIFIED


def test_cxsmiles_wd_different_pairs():
    """Different wD atom pairs produce different molecular representations."""
    import numpy as np

    from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer

    cxsmiles_1 = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl |wD:14.16|"
    cxsmiles_2 = "C1=C(C2=CC(Cl)=CC(F)=C2)C(F)=CC=C1CCCCC |wD:2.9|"

    mol_1 = make_mol(cxsmiles_1)
    mol_2 = make_mol(cxsmiles_2)

    mg_featurizer = SimpleMoleculeMolGraphFeaturizer()
    graph_1 = mg_featurizer(mol_1)
    graph_2 = mg_featurizer(mol_2)

    sum_1 = graph_1.V.sum(axis=0)
    sum_2 = graph_2.V.sum(axis=0)
    assert not np.array_equal(sum_1, sum_2)
