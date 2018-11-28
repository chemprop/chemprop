from argparse import Namespace
from typing import List, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

from chemprop.features.functional_groups import get_num_functional_groups, FunctionalGroupFeaturizer

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}


def get_atom_fdim(args: Namespace) -> int:
    """Gets the dimensionality of atom features."""
    if 'functional_group' in args.additional_atom_features:
        return ATOM_FDIM + get_num_functional_groups(args)
    return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    """Gets the dimensionality of bond features."""
    return BOND_FDIM + \
           args.virtual_edges * (1 + len(PATH_DISTANCE_BINS)) + \
           args.three_d * (1 + len(THREE_D_DISTANCE_BINS))


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


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :return: A PyTorch tensor containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond,
                  distance_path: int = None,
                  distance_3d: int = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :param distance_path: The topological (path) distance between the atoms in the bond.
    Note: This is always 1 if the atoms are actually bonded and >1 for "virtual" edges between non-bonded atoms.
    :param distance_3d: The bin index of the 3D distance in THREE_D_DISTANCE_BINS.
    :return: A PyTorch tensor containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
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

    if distance_path is not None:
        fbond += onek_encoding_unk(distance_path, PATH_DISTANCE_BINS)

    if distance_3d is not None:
        fbond += onek_encoding_unk(distance_3d, THREE_D_DISTANCE_BINS)

    return fbond


class MolGraph:
    def __init__(self, smiles: str, args: Namespace):
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond

        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)

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
                distances_3d = np.digitize(distances_3d, THREE_D_DISTANCE_BINS)  # bin 3d distances
            except:
                # zero distance matrix, in case rdkit errors out
                print('distance embedding failed')
                distances_3d = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))

        # Get topological (i.e. path-length) distance matrix and number of atoms
        distances_path = Chem.GetDistanceMatrix(mol)
        self.n_atoms = mol.GetNumAtoms()
        
        # Get atom features
        if 'functional_group' in args.additional_atom_features:
            fg_featurizer = FunctionalGroupFeaturizer(args)
            fg_features = fg_featurizer.featurize(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            if 'functional_group' in args.additional_atom_features:
                self.f_atoms.append(atom_features(atom, fg_features[i].tolist()))
            else:
                self.f_atoms.append(atom_features(atom))
            self.a2b.append([])

        if args.learn_virtual_edges:
            model = args.lve_model  # this is the MPN model, added to args so we can access it here
            f_atoms_cuda = torch.Tensor(self.f_atoms).cuda()
            processed_f_atoms_cuda = model.lve(f_atoms_cuda)
            lve_scores = torch.matmul(processed_f_atoms_cuda, f_atoms_cuda.t())
            symmetric_lve_scores = lve_scores + lve_scores.t()

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                # Randomly drop O(n_atoms) virtual edges so a total of O(n_atoms) edges instead of O(n_atoms^2)
                if bond is None:
                    if not args.virtual_edges:
                        continue

                    if args.drop_virtual_edges and hash(str((a1, a2))) % self.n_atoms != 0:
                        continue

                    # this option below doesn't seem to be as good
                    # if args.drop_virtual_edges and (hash(mol_fatoms[a1])+hash(mol_fatoms[a2])) % n_atoms != 0:
                    #     continue

                    if args.learn_virtual_edges:
                        # if score less than 0, don't add the edge
                        model = args.lve_model
                        if symmetric_lve_scores[a1, a2] < 0:  # want symmetry in a1/a2
                            continue

                distance_3d = distances_3d[a1, a2] if args.three_d else None
                distance_path = distances_path[a1, a2] if args.virtual_edges else None

                f_bond = bond_features(bond, distance_path=distance_path, distance_3d=distance_3d)
                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


class BatchMolGraph:
    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim(args)
        self.bond_fdim = self.atom_fdim + get_bond_fdim(args)

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(len(in_bonds) for in_bonds in a2b)

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self):
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def bert_mask(self,
                  mask: List[int],
                  zero_prob: float = 0.8,
                  diff_prob: float = 0.1,
                  same_prob: float = 0.1):
        """
        Masks the atom features for bert pretraining.

        :param mask: A list of 0s and 1s where 0s indicate which atoms to mask out.
        :param zero_prob: The probability of replacing a masked atom with 0s.
        :param diff_prob: The probability of replacing a masked atom with a different atom feature vector.
        :param same_prob: The probability of keeping a masked atom's feature vector.
        """
        f_atoms = self.f_atoms[1:]  # 1 for padding

        assert zero_prob + diff_prob + same_prob == 1
        assert len(mask) == len(f_atoms)

        mask_indices = np.where(np.array(mask) == 0)[0]
        rands = np.random.random(len(mask_indices))

        for i, rand in zip(mask_indices, rands):
            if rand < zero_prob:
                f_atoms[i] = 0  # mask with zeros
            elif rand < zero_prob + diff_prob:
                f_atoms[i] = f_atoms[np.random.randint(len(f_atoms))]  # mask with different atom feature vector
            # mask with same atom feature vector (i.e. do nothing)


def mol2graph(smiles_batch: List[str],
              args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, args)
            # Memoize if we're not chunking or learning virtual edges, to save memory
            if ((args.num_chunks == 1 and args.prespecified_chunk_dir is None) or args.memoize_chunks) and not args.learn_virtual_edges:
                SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)
    
    return BatchMolGraph(mol_graphs, args)
