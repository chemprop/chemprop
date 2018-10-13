from argparse import Namespace
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

# Atom feature sizes
ATOM_FEATURES = {
    'atomic_num': list(range(100)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'implicit_valence': [0, 1, 2, 3, 4, 5, 6],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    'num_radical_electrons': [0, 1, 2]
}

# len(choices) + 1 to include room for uncommon values; + 1 at end for IsAromatic
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 1
BOND_FDIM = 14

# Memoization
SMILES_TO_FEATURES = dict()


class DummyConformer:  # for use when 3d embedding fails
    def GetAtomPosition(self, id: int) -> List[int]:
        return [0, 0, 0]


def get_atom_fdim(args: Namespace) -> int:
    """Gets the dimensionality of atom features."""
    return ATOM_FDIM + 3 * args.three_d


def get_bond_fdim(args: Namespace) -> int:
    """Gets the dimensionality of bond features."""
    return BOND_FDIM + args.three_d + (11 * args.virtual_edges)


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, atom_position: List[float] = None) -> torch.Tensor:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param atom_position: A length-3 list containing the xyz coordinates of the atom.
    :return: A PyTorch tensor containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) \
               + onek_encoding_unk(atom.GetDegree(), ATOM_FEATURES['degree']) \
               + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) \
               + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) \
               + onek_encoding_unk(int(atom.GetImplicitValence()), ATOM_FEATURES['implicit_valence']) \
               + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) \
               + onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) \
               + onek_encoding_unk(int(atom.GetNumRadicalElectrons()), ATOM_FEATURES['num_radical_electrons']) \
               + [atom.GetIsAromatic()]
    if atom_position is not None:
        features += atom_position
    return torch.Tensor(features)


def bond_features(bond: Chem.rdchem.Bond, distance_path: int = None, distance_3d: float = None) -> torch.Tensor:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :param distance_path: The topological (path) distance between the atoms in the bond.
    Note: This is always 1 if the atoms are actually bonded and >1 for "virtual" edges between non-bonded atoms.
    :param distance_3d: The Euclidean distance between the atoms in 3D space.
    :return: A PyTorch tensor containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * 13
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    fdistance_path = onek_encoding_unk(distance_path, list(range(10))) if distance_path is not None else []
    fdistance_3d = [distance_3d] if distance_3d is not None else []

    return torch.Tensor(fbond + fdistance_path + fdistance_3d)


def mol2graph(mol_batch: List[str], args: Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Converts a list of SMILES strings to a batch of molecular graphs consisting of of PyTorch tensors.

    :param mol_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A tuple of tensors representing a batch of molecular graphs with the following elements:
    1) fatoms: A tensor of shape (num_atoms, atom_fdim) containing the atom features.
    2) fbonds: A tensor of shape (num_bonds, bond_fdim) containing the bond features.
    3) agraph: A tensor of shape (num_atoms, max_num_bonds) where agraph[a] contains the indices of atoms bonded to a.
    4) bgraph: A tensor of shape (num_bonds, max_num_bonds) where bgraph[b] contains the indices of the bonds that
    start where bond b ends. (Ex. If b = (x, y) then bgraph[b] contains all bonds (y, z) for any atom z bonded to y.)
    5) ascope: A list of tuples of length batch_size where each tuple is (start, size) where start is the index of
    the first atom in a given molecular graph and size is the number of atoms in that graph.
    6) bscope: Similar to ascope but for bond indices rather than atom indices.
    """
    padding = torch.zeros(get_atom_fdim(args) + get_bond_fdim(args))
    fatoms, fbonds = [], [padding]  # Ensure bond is 1-indexed
    in_bonds, all_bonds = [], [(-1, -1)]  # Ensure bond is 1-indexed
    ascope = []
    bscope = []
    total_atoms = 0
    total_bonds = 1

    for smiles in mol_batch:
        if smiles in SMILES_TO_FEATURES:
            mol_fatoms, mol_fbonds, mol_all_bonds, n_atoms, n_bonds = SMILES_TO_FEATURES[smiles]
        else:
            mol_fatoms, mol_fbonds, mol_all_bonds = [], [], []

            # Convert smiles to molecule
            mol = Chem.MolFromSmiles(smiles)

            if args.addHs:
                mol = Chem.AddHs(mol)

            # Get 3D distance matrix
            if args.three_d:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                if not args.addHs:
                    mol = Chem.RemoveHs(mol)
                try:
                    distances_3d = Chem.Get3DDistanceMatrix(mol)
                    conformer = mol.GetConformer()
                except:
                    # zero distance matrix, in case rdkit errors out
                    print('distance embedding failed')
                    distances_3d = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))
                    conformer = DummyConformer()

            # Get topological (i.e. path-length) distance matrix and number of atoms
            distances_path = Chem.GetDistanceMatrix(mol)
            n_atoms = mol.GetNumAtoms()

            # Get atom features
            for atom in mol.GetAtoms():
                atom_position = list(conformer.GetAtomPosition(atom.GetIdx())) if args.three_d else None
                mol_fatoms.append(atom_features(atom, atom_position))

            n_bonds = 0
            # Get bond features
            for a1 in range(n_atoms):
                for a2 in range(a1 + 1, n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None and not args.virtual_edges:
                        continue

                    distance_3d = distances_3d[a1, a2] if args.three_d else None
                    distance_path = distances_path[a1, a2] if args.virtual_edges else None

                    mol_all_bonds.append((a1, a2))
                    mol_fbonds.append(torch.cat([mol_fatoms[a1], bond_features(bond,
                                                                               distance_path=distance_path,
                                                                               distance_3d=distance_3d)], dim=0))

                    mol_all_bonds.append((a2, a1))
                    mol_fbonds.append(torch.cat([mol_fatoms[a2], bond_features(bond,
                                                                               distance_path=distance_path,
                                                                               distance_3d=distance_3d)], dim=0))
                    n_bonds += 2
            # Memoize if we're not chunking to save memory
            SMILES_TO_FEATURES[smiles] = (mol_fatoms, mol_fbonds, mol_all_bonds, n_atoms, n_bonds)

        # Add molecule features to batch features
        fatoms.extend(mol_fatoms)
        fbonds.extend(mol_fbonds)

        in_bonds.extend([[] for _ in range(n_atoms)])

        for a1, a2 in mol_all_bonds:
            a1 += total_atoms
            a2 += total_atoms

            bond_idx = len(all_bonds)
            all_bonds.append((a1, a2))
            in_bonds[a2].append(bond_idx)

        ascope.append((total_atoms, n_atoms))
        bscope.append((total_bonds, n_bonds))
        total_atoms += n_atoms
        total_bonds += n_bonds

    max_num_bonds = max(len(bonds) for bonds in in_bonds)

    fatoms = torch.stack(fatoms, dim=0)
    fbonds = torch.stack(fbonds, dim=0)

    # Map each atom to all bonds going into that atom
    agraph = torch.LongTensor([bonds + [0] * (max_num_bonds - len(bonds)) for bonds in in_bonds])  # zero padding

    # Map each bond to all bonds going into that bond's start atom
    bgraph = [[]] + [[bond if all_bonds[bond][0] != a2 else 0 for bond in in_bonds[a1]] for a1, a2 in all_bonds[1:]]
    bgraph = torch.LongTensor([bonds + [0] * (max_num_bonds - len(bonds)) for bonds in bgraph])  # zero padding

    return fatoms, fbonds, agraph, bgraph, ascope, bscope
