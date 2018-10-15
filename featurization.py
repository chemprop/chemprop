from abc import ABC, abstractmethod
from argparse import Namespace
from collections import defaultdict
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
SMILES_TO_GRAPH = {}


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


class MolGraph:
    def __init__(self, smile: str, args: Namespace):
        # Note: Bond features are undirected even though bonds are directed, so two indices map to each bond feature
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = {}  # mapping from bond index to bond features (dict b/c two indices map to each feature)
        self.a2a = defaultdict(list)  # mapping from atom index to neighboring atom indices
        self.a2b = defaultdict(list)  # mapping from atom index to incoming bond indices
        self.b2a = {}  # mapping from bond index to tuple of bonded atom indices (out_atom, in_atom)
        self.f_bonds_with_atom = []  # mapping from bond index to concat(bond, in_atom) features
        self.b2b = {}  # mapping from bond index to indices of bonds going into atom this bond comes from

        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smile)

        # Add hydrogens
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
        self.n_atoms = mol.GetNumAtoms()

        # Get atom features
        for atom in mol.GetAtoms():
            atom_position = list(conformer.GetAtomPosition(atom.GetIdx())) if args.three_d else None
            self.f_atoms.append(atom_features(atom, atom_position))

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                # Randomly drop O(n_atoms) virtual edges so a total of O(n_atoms) edges instead of O(n_atoms^2)
                if bond is None:
                    if not args.virtual_edges:
                        continue

                    if args.drop_virtual_edges and hash(str(a2)) % self.n_atoms != 0:
                        continue

                distance_3d = distances_3d[a1, a2] if args.three_d else None
                distance_path = distances_path[a1, a2] if args.virtual_edges else None

                # Get bond feature (note: two indices b/c directed edges)
                b1 = self.n_bonds
                b2 = self.n_bonds + 1
                f_bond = bond_features(bond, distance_path=distance_path, distance_3d=distance_3d)
                self.f_bonds[b1] = f_bond
                self.f_bonds[b2] = f_bond
                self.n_bonds += 2

                # Update index mappings
                self.a2a[a1].append(a2)
                self.a2a[a2].append(a1)
                self.a2b[a1].append(b1)
                self.a2b[a2].append(b2)
                self.b2a[b1] = (a2, a1)
                self.b2a[b2] = (a1, a2)

        # Update bond-to-bond mapping and combine bond and atom features for f_bond_with_atom
        for b in range(self.n_bonds):
            a1, a2 = self.b2a[b]
            in_bonds = self.a2b[a1]
            self.b2b[b] = [bond for bond in in_bonds if self.b2a[b][1] != a1]  # leave out reverse of bond
            self.f_bonds_with_atom.append(torch.cat([self.f_atoms[a1], self.f_bonds[b]], dim=0))


class BatchMolGraph(ABC):
    def __init__(self, mol_graphs: List[MolGraph]):
        self.mol_graphs = mol_graphs


class BatchMolGraphAtom(BatchMolGraph):
    def __init__(self, mol_graphs: List[MolGraph]):
        super(BatchMolGraphAtom, self).__init__(mol_graphs)


class BatchMolGraphBond(BatchMolGraph):
    def __init__(self, mol_graphs: List[MolGraph]):
        super(BatchMolGraphBond, self).__init__(mol_graphs)

        self.atom_fdim = len(mol_graphs[0].f_atoms[0])
        self.bond_fdim = len(mol_graphs[0].f_bonds_with_atom[0])

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.ascope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.bscope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        f_atoms = []  # atom features
        f_bonds = [torch.zeros(self.bond_fdim)]  # combined atom/bond features
        a2b = {}  # mapping from atom index to incoming bond indices
        b2b = {0: []}  # mapping from bond index to incoming bond indices (index 0 for padding)
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds_with_atom)

            for atom in range(mol_graph.n_atoms):
                a2b[self.n_atoms + atom] = mol_graph.a2b[atom]

            for bond in range(mol_graph.n_bonds):
                b2b[self.n_bonds + bond] = mol_graph.b2b[bond]

            self.ascope.append((self.n_atoms, mol_graph.n_atoms))
            self.bscope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(len(in_bonds) for in_bonds in b2b.values())

        self.f_atoms = torch.cat(f_atoms, dim=0)
        self.f_bonds = torch.cat(f_bonds, dim=0)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2b = torch.LongTensor([b2b[b] + [0] * (self.max_num_bonds - len(b2b[b])) for b in range(self.n_bonds)])


def mol2graph(smiles: List[str], args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smile in smiles:
        if smile not in SMILES_TO_GRAPH:
            SMILES_TO_GRAPH[smile] = MolGraph(smile, args)
        mol_graphs.append(SMILES_TO_GRAPH[smile])

    if args.message_type == 'bond':
        return BatchMolGraphBond(mol_graphs)
    elif args.message_type == 'atom':
        return BatchMolGraphAtom(mol_graphs)
    else:
        raise ValueError('Message type "{}" not supported.'.format(args.message_type))
