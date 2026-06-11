#!/usr/bin/env python3
"""
Test: can we hack Chemprop to distinguish atropisomers by manually setting
chiral tags on the sp² carbons at the atropisomer axis?

The idea: parse CXSMILES wD/wU tags, identify the atropisomeric bond,
manually set @/@@ on those atoms, and see if Chemprop's atom featurizer
produces different features.
"""

import sys
import re
from pathlib import Path

from rdkit import Chem
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.atom import MultiHotAtomFeaturizer


def parse_wd_wu(cxsmiles: str):
    """Extract wD/wU constraints from CXSMILES."""
    if " |" not in cxsmiles:
        return None, []
    smi, metadata = cxsmiles.split(" |", 1)
    metadata = metadata.strip("||")
    constraints = []
    for match in re.finditer(r'(wD|wU):(\d+)\.(\d+)', metadata):
        kind = match.group(1)  # wD or wU
        a, b = int(match.group(2)), int(match.group(3))
        constraints.append((kind, a, b))
    return smi, constraints


def set_atropisomer_chiral_tags(mol: Chem.Mol, kind: str, atom_a: int, atom_b: int):
    """
    Manually set chiral tags on the two atoms of the atropisomeric bond.

    For wD (wedge down): atom_a is DOWN, atom_b is UP
    For wU (wedge unspecified): both atoms get same tag (no distinction)

    This is a CHEMICAL HACK - these are sp² carbons, not tetrahedral centers.
    RDKit may not allow this, or may ignore it.
    """
    atom_a_obj = mol.GetAtomWithIdx(atom_a)
    atom_b_obj = mol.GetAtomWithIdx(atom_b)

    if kind == "wD":
        # DOWN vs UP → different chiral tags
        # CHI_TETRAHEDRAL_CW = 1, CHI_TETRAHEDRAL_CCW = 2
        atom_a_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        atom_b_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    elif kind == "wU":
        # Unspecified → no distinction (both unset)
        atom_a_obj.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom_b_obj.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    return mol


def get_atom_features(mol: Chem.Mol, featurizer):
    """Get atom features from Chemprop's featurizer."""
    features = []
    for atom in mol.GetAtoms():
        feat = featurizer(atom)
        features.append(feat)
    return np.array(features)


def test_atropisomer_hack():
    """Test the atropisomer chiral tag hack."""
    print("=== Testing Atropisomer Chiral Tag Hack ===\n")

    # Lines 49-50: same SMILES, wD vs wU
    cxsmi_wd = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl |wD:14.16|"
    cxsmi_wu = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl |wU:14.16|"

    smi, constraints_wd = parse_wd_wu(cxsmi_wd)
    smi2, constraints_wu = parse_wd_wu(cxsmi_wu)

    print(f"Base SMILES: {smi}")
    print(f"wD constraints: {constraints_wd}")
    print(f"wU constraints: {constraints_wu}")
    print()

    # Parse with RDKit
    mol_wd = Chem.MolFromSmiles(smi)
    mol_wu = Chem.MolFromSmiles(smi2)

    # Create featurizer
    featurizer = MultiHotAtomFeaturizer.v1()

    # Before hack: identical
    feat_wd_before = get_atom_features(mol_wd, featurizer)
    feat_wu_before = get_atom_features(mol_wu, featurizer)
    print(f"Before hack - atom features identical: {np.array_equal(feat_wd_before, feat_wu_before)}")
    print()

    # Apply hack
    print("Applying chiral tag hack...")
    for kind, a, b in constraints_wd:
        print(f"  Setting chiral tags on bond {a}-{b} (kind={kind})")
        try:
            mol_wd = set_atropisomer_chiral_tags(mol_wd, kind, a, b)
            atom_a = mol_wd.GetAtomWithIdx(a)
            atom_b = mol_wd.GetAtomWithIdx(b)
            print(f"  Atom {a}: chiral tag = {atom_a.GetChiralTag()}")
            print(f"  Atom {b}: chiral tag = {atom_b.GetChiralTag()}")
        except Exception as e:
            print(f"  ERROR: {e}")
            return

    for kind, a, b in constraints_wu:
        print(f"  Setting chiral tags on bond {a}-{b} (kind={kind})")
        try:
            mol_wu = set_atropisomer_chiral_tags(mol_wu, kind, a, b)
            atom_a = mol_wu.GetAtomWithIdx(a)
            atom_b = mol_wu.GetAtomWithIdx(b)
            print(f"  Atom {a}: chiral tag = {atom_a.GetChiralTag()}")
            print(f"  Atom {b}: chiral tag = {atom_b.GetChiralTag()}")
        except Exception as e:
            print(f"  ERROR: {e}")
            return

    print()

    # After hack: different?
    feat_wd_after = get_atom_features(mol_wd, featurizer)
    feat_wu_after = get_atom_features(mol_wu, featurizer)
    print(f"After hack - atom features identical: {np.array_equal(feat_wd_after, feat_wu_after)}")
    print()

    if not np.array_equal(feat_wd_after, feat_wu_after):
        # Find which atoms differ
        diff_mask = feat_wd_after != feat_wu_after
        diff_atoms = np.where(diff_mask.any(axis=1))[0]
        print(f"Differing atom indices: {diff_atoms}")
        for atom_idx in diff_atoms:
            print(f"  Atom {atom_idx}:")
            print(f"    wD features: {feat_wd_after[atom_idx]}")
            print(f"    wU features: {feat_wu_after[atom_idx]}")
        print()

        # Check MolGraphs
        mg_featurizer = SimpleMoleculeMolGraphFeaturizer()
        graph_wd = mg_featurizer(mol_wd)
        graph_wu = mg_featurizer(mol_wu)
        print(f"MolGraph V identical: {np.array_equal(graph_wd.V, graph_wu.V)}")
        print(f"MolGraph E identical: {np.array_equal(graph_wd.E, graph_wu.E)}")
        print(f"MolGraph edge_index identical: {np.array_equal(graph_wd.edge_index, graph_wu.edge_index)}")
    else:
        print("Hack FAILED - atom features are still identical!")

    # Check what Chemprop's atom featurizer actually captures for chiral tags
    print()
    print("=== Chemprop Atom Featurizer Chiral Tag Handling ===")
    print(f"Chiral tag choices: {featurizer.chiral_tags}")
    print(f"  0 (UNSPECIFIED): {'included' if 0 in featurizer.chiral_tags else 'NOT included'}")
    print(f"  1 (CW): {'included' if 1 in featurizer.chiral_tags else 'NOT included'}")
    print(f"  2 (CCW): {'included' if 2 in featurizer.chiral_tags else 'NOT included'}")
    print(f"  3 (TETRAHEDRAL): {'included' if 3 in featurizer.chiral_tags else 'NOT included'}")


def test_smiles_roundtrip():
    """Test if chiral tags survive SMILES round-trip."""
    print("\n=== SMILES Round-Trip Test ===\n")

    smi = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl"
    mol = Chem.MolFromSmiles(smi)

    # Set chiral tags on atoms 14 and 16
    mol.GetAtomWithIdx(14).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    mol.GetAtomWithIdx(16).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)

    print(f"Before round-trip:")
    print(f"  Atom 14: {mol.GetAtomWithIdx(14).GetChiralTag()}")
    print(f"  Atom 16: {mol.GetAtomWithIdx(16).GetChiralTag()}")

    # Convert to SMILES
    smi_out = Chem.MolToSmiles(mol, isomericSmiles=True)
    print(f"  Output SMILES: {smi_out[:100]}")

    # Parse back
    mol2 = Chem.MolFromSmiles(smi_out)
    print(f"\nAfter round-trip:")
    print(f"  Atom 14: {mol2.GetAtomWithIdx(14).GetChiralTag()}")
    print(f"  Atom 16: {mol2.GetAtomWithIdx(16).GetChiralTag()}")

    if mol2.GetAtomWithIdx(14).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
        print("\n  Tags LOST during round-trip!")
        print("  This means we MUST set tags on the Mol object, not in SMILES.")
    else:
        print("\n  Tags SURVIVED round-trip!")


def test_integration_with_chemprop():
    """Test end-to-end: CXSMILES → hacked Mol → Chemprop prediction."""
    print("\n=== End-to-End Test ===\n")

    cxsmi_wd = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl |wD:14.16|"
    cxsmi_wu = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl |wU:14.16|"

    smi_wd, constraints_wd = parse_wd_wu(cxsmi_wd)
    smi_wu, constraints_wu = parse_wd_wu(cxsmi_wu)

    mol_wd = Chem.MolFromSmiles(smi_wd)
    mol_wu = Chem.MolFromSmiles(smi_wu)

    # Apply hack
    for kind, a, b in constraints_wd:
        set_atropisomer_chiral_tags(mol_wd, kind, a, b)
    for kind, a, b in constraints_wu:
        set_atropisomer_chiral_tags(mol_wu, kind, a, b)

    # Get MolGraphs
    mg_featurizer = SimpleMoleculeMolGraphFeaturizer()
    graph_wd = mg_featurizer(mol_wd)
    graph_wu = mg_featurizer(mol_wu)

    # Compare
    print(f"Atom features (V):")
    print(f"  wD shape: {graph_wd.V.shape}, wU shape: {graph_wu.V.shape}")
    print(f"  Identical: {np.array_equal(graph_wd.V, graph_wu.V)}")

    if not np.array_equal(graph_wd.V, graph_wu.V):
        diff_atoms = np.where((graph_wd.V != graph_wu.V).any(axis=1))[0]
        print(f"  Differing atoms: {diff_atoms}")

        # Compute the molecular representation (sum aggregation)
        sum_wd = graph_wd.V.sum(axis=0)
        sum_wu = graph_wu.V.sum(axis=0)
        print(f"\n  Sum-aggregated V (molecular representation):")
        print(f"    wD: {sum_wd}")
        print(f"    wU: {sum_wu}")
        print(f"    Identical: {np.array_equal(sum_wd, sum_wu)}")
    else:
        print("  Atom features are identical - hack did NOT work")

    print()
    print("Bond features (E):")
    print(f"  Identical: {np.array_equal(graph_wd.E, graph_wu.E)}")
    print(f"  Edge indices identical: {np.array_equal(graph_wd.edge_index, graph_wu.edge_index)}")


if __name__ == "__main__":
    test_atropisomer_hack()
    test_smiles_roundtrip()
    test_integration_with_chemprop()
