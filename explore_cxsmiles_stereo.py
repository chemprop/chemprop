#!/usr/bin/env python3
"""
Deep dive: can we extract atropisomer info from CXSMILES that RDKit strips?

Key hypothesis: wD/wU in this file encode bond stereochemistry (wedge down/up)
on the single bond that connects two aromatic rings - the atropisomer axis.
wD = wedge down, wU = wedge unspecified.

We need to understand:
1. What wD/wU actually means in this CXSMILES dialect
2. Whether RDKit preserves ANY of this info during parsing
3. Whether we can encode this as bond stereochemistry in the graph
"""

import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).parent))


def analyze_atropisomer_bond():
    """Analyze the bond that connects two aromatic rings in an atropisomer."""
    print("=== Atropisomer Bond Analysis ===\n")

    # Lines 49-50: C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl
    # wD:14.16 vs wU:14.16
    #
    # The two aromatic rings are connected by a single bond.
    # In this SMILES, ring 1 is C1(Cl)=C2...C=C1, ring 2 is ...C1=C2...C=C1Cl
    # The atoms 14 and 16 are in different rings.
    #
    # Let me trace through the SMILES to find atoms 14 and 16:
    smi = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl"
    mol = Chem.MolFromSmiles(smi)

    print(f"SMILES: {smi}")
    print(f"Num atoms: {mol.GetNumAtoms()}")
    print()

    # Find which atoms are in which rings
    ri = mol.GetRingInfo()
    ring_atoms = {}
    for i, atom in enumerate(mol.GetAtoms()):
        rings = ri.AtomRings(i)
        ring_atoms[i] = rings

    print("Ring membership:")
    for i, rings in sorted(ring_atoms.items()):
        atom = mol.GetAtomWithIdx(i)
        print(f"  Atom {i:2d} ({atom.GetSymbol()}): {len(rings)} ring(s)")

    print()
    print("Bonds between ring atoms:")
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_rings = ring_atoms[begin]
        end_rings = ring_atoms[end]
        if begin_rings and end_rings:
            # Both atoms are in rings
            shared_rings = set(begin_rings) & set(end_rings)
            if not shared_rings:
                # Atoms are in rings but not shared rings - this is a ring-connecting bond
                print(f"  Bond {begin}-{end} ({bond.GetBondType()}): connects different rings "
                      f"(atropisomer axis candidate)")
            else:
                print(f"  Bond {begin}-{end} ({bond.GetBondType()}): within same ring(s)")

    print()

    # The wD:14.16 tag - atoms 14 and 16
    # Let me check what bond connects atoms 14 and 16
    print("Atoms 14 and 16:")
    for i in [14, 16]:
        atom = mol.GetAtomWithIdx(i)
        print(f"  Atom {i}: {atom.GetSymbol()}, rings={ring_atoms[i]}, "
              f"neighbors={[n.GetIdx() for n in atom.GetNeighbors()]}")

    # Check if there's a bond between 14 and 16
    bond_14_16 = mol.GetBondBetweenAtoms(14, 16)
    if bond_14_16:
        print(f"\n  Direct bond 14-16: type={bond_14_16.GetBondType()}, "
              f"stereo={bond_14_16.GetStereo()}, dir={bond_14_16.GetBondDir()}")
    else:
        print(f"\n  No direct bond between atoms 14 and 16")
        # They might be connected via a path
        print("  Checking path between 14 and 16...")
        for atom_idx in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(atom_idx).HasProp('_CIPCode'):
                print(f"    Atom {atom_idx} has CIP code: {mol.GetAtomWithIdx(atom_idx).GetProp('_CIPCode')}")

    print()


def test_cxsmiles_info_parsing():
    """Check if RDKit's CXSmilesInfo captures anything during parsing."""
    print("=== CXSmilesInfo Parsing ===\n")

    cxsmi = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl |wD:14.16|"
    smi = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl"

    # Standard parse
    mol = Chem.MomFromSmiles(smi) if hasattr(Chem, 'MolFromSmiles') else None
    mol = Chem.MolFromSmiles(smi)

    # Check all properties
    print(f"Mol from SMILES (no CXSMILES metadata):")
    print(f"  Properties: {mol.GetPropNames()}")
    print(f"  Num conformers: {mol.GetNumConformers()}")

    # Check if CXSmilesInfo is available
    params = Chem.SmilesParserParams()
    print(f"\nSmilesParserParams attributes: {[a for a in dir(params) if not a.startswith('_')]}")

    # Try with full CXSMILES
    mol_cx = Chem.MolFromSmiles(cxsmi)
    print(f"\nMol from CXSMILES (with |wD:14.16|):")
    print(f"  Properties: {mol_cx.GetPropNames()}")
    print(f"  Num conformers: {mol_cx.GetNumConformers()}")
    print(f"  Are mols identical: {Chem.MolToSmiles(mol) == Chem.MolToSmiles(mol_cx)}")

    # Check bond stereo on all bonds
    print(f"\nAll bonds and their stereo/direction:")
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE or bond.GetBondDir() != Chem.BondDir.NONE:
            print(f"  Bond {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()}: "
                  f"stereo={bond.GetStereo()}, dir={bond.GetBondDir()}")

    print()


def test_manual_stereo_encoding():
    """Try manually setting bond stereochemistry on the atropisomer axis."""
    print("=== Manual Bond Stereo Encoding ===\n")

    smi = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl"
    mol = Chem.MolFromSmiles(smi)

    # Find the single bond connecting the two aromatic rings
    ri = mol.GetRingInfo()
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if bond.GetBondType() == Chem.BondType.SINGLE:
            begin_rings = ri.AtomRings(begin)
            end_rings = ri.AtomRings(end)
            if begin_rings and end_rings and not (set(begin_rings) & set(end_rings)):
                print(f"Found ring-connecting single bond: {begin}-{end}")
                print(f"  Current stereo: {bond.GetStereo()}")

                # Try setting bond stereo
                # Note: RDKit uses BondStereo for double bonds, not single bonds
                # For single bonds, we'd need to use BondDir (wedge/dash)
                # But BondDir is for 2D rendering, not 3D stereochemistry
                bond.SetBondDir(Chem.BondDir.ENDUPWARDS)  # wedge up
                print(f"  After SetBondDir(ENDUPWARDS): {bond.GetBondDir()}")

                # Check if this survives round-trip through SMILES
                smi_up = Chem.MolToSmiles(mol, isomericSmiles=True)
                print(f"  SMILES after: {smi_up[:80]}")

                # Reset
                bond.SetBondDir(Chem.BondDir.NONE)
                mol2 = Chem.MolFromSmiles(smi)
                bond2 = mol2.GetBondBetweenAtoms(begin, end)
                bond2.SetBondDir(Chem.BondDir.ENDDOWNWARDS)  # wedge down
                smi_down = Chem.MolToSmiles(mol2, isomericSmiles=True)
                print(f"  SMILES after DOWN: {smi_down[:80]}")

                # Are they different?
                if smi_up != smi_down:
                    print(f"  SUCCESS: Different SMILES produced!")
                else:
                    print(f"  FAIL: Same SMILES (BondDir is rendering-only, not stereo)")

    print()


def test_alternative_approaches():
    """Explore alternative ways to encode atropisomerism."""
    print("=== Alternative Approaches ===\n")

    smi = "C1(Cl)=C2C(F)=CC=C1CCCCCC1=C2C=C(F)C=C1Cl"
    mol = Chem.MolFromSmiles(smi)

    # Approach 1: Use atom-level properties to mark the atropisomer state
    print("1. Atom-level property marking:")
    mol.SetProp("atropisomer", "wD:14.16")
    can = Chem.MolToSmiles(mol, isomericSmiles=True)
    print(f"   Canonical: {can[:80]}")
    print(f"   Properties preserved: {mol.GetPropNames()}")
    print()

    # Approach 2: Use a custom featurizer that extracts wD/wU from CXSMILES
    print("2. Custom CXSMILES featurizer approach:")
    print("   Parse wD/wU from CXSMILES metadata")
    print("   Extract the atropisomer axis bond (atoms from wD/wU)")
    print("   Compute dihedral angle from 3D conformer")
    print("   Add as extra atom/bond feature to MolGraph")
    print()

    # Approach 3: Convert to 3D conformer and use conformation-dependent descriptors
    print("3. 3D conformer descriptors:")
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_3d)
    conf = mol_3d.GetConformer()

    # Get positions of atoms 14 and 16 and their ring partners
    pos_14 = conf.GetAtomPosition(14)
    pos_16 = conf.GetAtomPosition(16)
    print(f"   Atom 14 position: ({pos_14.x:.3f}, {pos_14.y:.3f}, {pos_14.z:.3f})")
    print(f"   Atom 16 position: ({pos_16.x:.3f}, {pos_16.y:.3f}, {pos_16.z:.3f})")

    # Compute dihedral angle between the two ring planes
    # Need 3+ atoms per ring to define a plane
    ri = mol_3d.GetRingInfo()
    rings = ri.AtomRings(14)  # rings containing atom 14
    if rings:
        ring_atoms_14 = mol_3d.GetRingInfo().AtomRings(14)
        # Map back to original mol (after AddHs, indices may shift)
        print(f"   Atom 14 is in ring(s): {ring_atoms_14}")

    print()

    # Approach 4: Force different atom ordering that creates different canonical SMILES
    print("4. Atom reordering approach:")
    print("   If we could reorder atoms such that the atropisomer state is")
    print("   reflected in the canonical SMILES, Chemprop would distinguish them.")
    print("   But canonical SMILES is topology-only (no 3D info), so this won't work.")
    print("   The atropisomer difference is purely conformational (3D).")
    print()


def test_wd_wu_meaning():
    """Understand what wD vs wU actually encodes."""
    print("=== wD vs wU Meaning ===\n")

    # Looking at the data:
    # Lines 49-50: same SMILES, wD:14.16 vs wU:14.16
    # Lines 51-52: same SMILES, wD:2.9 vs wU:2.9
    #
    # wD and wU use the SAME atom pairs but different prefixes.
    # In CXSMILES context:
    # - wD = "wedge down" - the bond between these atoms is shown pointing DOWN
    # - wU = "wedge unspecified" - the bond stereochemistry is NOT specified
    # - (there's also wU/wU variants and wU/wD)
    #
    # For atropisomers, wD vs wU indicates whether the atropisomeric configuration
    # is known (wD/wU) or unknown/unspecified (wU alone).
    #
    # BUT WAIT - looking at the data again:
    # Lines 49: wD:14.16 (wedge down)
    # Line 50: wU:14.16 (wedge unspecified)
    # These represent the SAME molecule but with different stereochem confidence.
    # wD says "this atropisomer has a DOWN configuration"
    # wU says "we're unsure about this atropisomer's configuration"
    #
    # NOT different atropisomers of the same molecule!
    # They're the same molecule with different metadata about stereochem certainty.

    print("KEY INSIGHT: wD vs wU does NOT mean different atropisomers!")
    print()
    print("wD = wedge DOWN (stereochemistry is specified as 'down')")
    print("wU = wedge UNSPECIFIED (stereochemistry is unknown/unspecified)")
    print()
    print("So lines 49 and 50 are the SAME molecule, just with different")
    print("stereochem certainty. They should produce the SAME prediction.")
    print()

    # The real atropisomer question: are there molecules in this file where
    # the SAME base SMILES has BOTH wD:atoms AND wU:atoms on different bonds?
    # That would indicate different atropisomeric configurations.

    cxsmi_file = Path(__file__).parent / "cleaned.cxsmi"
    with open(cxsmi_file) as f:
        lines = f.readlines()

    # Check for wD AND wU on the same molecule
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if " |" not in line:
            continue
        smi, meta = line.split(" |", 1)
        meta = meta.strip("||")

        has_wd = "wD:" in meta
        has_wu = "wU:" in meta

        if has_wd and has_wu:
            # Both wD and wU present - different bonds have different certainty
            print(f"Line {line_num}: Both wD and wU (mixed certainty):")
            print(f"  SMILES: {smi[:70]}...")
            print(f"  Meta: |{meta}|")
            print()


if __name__ == "__main__":
    test_wd_wu_meaning()
    print()
    analyze_atropisomer_bond()
    print()
    test_cxsmiles_info_parsing()
    test_manual_stereo_encoding()
    test_alternative_approaches()
