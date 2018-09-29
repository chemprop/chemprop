from argparse import Namespace
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn.functional as F
import torch.nn as nn

from nn_utils import index_select_ND

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
SMILES_TO_FEATURES = {}

class DummyConformer: #for use when 3d embedding fails
    def GetAtomPosition(self, id):
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


def mol2graph(mol_batch: List[str], args: Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Converts a list of SMILES strings to a batch of molecular graphs consisting of of PyTorch tensors.

    :param mol_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A tuple of tensors representing a batch of molecular graphs. TODO: Better explanation.
    """
    padding = torch.zeros(get_atom_fdim(args) + get_bond_fdim(args))
    fatoms, fbonds = [], [padding]  # Ensure bond is 1-indexed
    in_bonds, all_bonds = [], [(-1, -1)]  # Ensure bond is 1-indexed
    scope = []
    total_atoms = 0

    for smiles in mol_batch:
        if smiles in SMILES_TO_FEATURES:
            mol_fatoms, mol_fbonds, mol_all_bonds, n_atoms = SMILES_TO_FEATURES[smiles]
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

            # Memoize
            SMILES_TO_FEATURES[smiles] = (mol_fatoms, mol_fbonds, mol_all_bonds, n_atoms)

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

        scope.append((total_atoms, n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    max_num_bonds = max(len(bonds) for bonds in in_bonds)

    fatoms = torch.stack(fatoms, dim=0)
    fbonds = torch.stack(fbonds, dim=0)
    agraph = torch.zeros(total_atoms, max_num_bonds, dtype=torch.long)
    bgraph = torch.zeros(total_bonds, max_num_bonds, dtype=torch.long)

    for a in range(total_atoms):
        for i, b in enumerate(in_bonds[a]):
            agraph[a, i] = b

    for b1 in range(1, total_bonds):
        x, y = all_bonds[b1]
        for i, b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1, i] = b2

    return fatoms, fbonds, agraph, bgraph, scope


def build_MPN(num_tasks: int, args: Namespace) -> nn.Module:
    """
    Builds a message passing neural network including final linear layers and initializes parameters.

    :param num_tasks: Number of tasks to predict.
    :param args: Arguments.
    :return: An nn.Module containing the MPN encoder along with final linear layers with parameters initialized.
    """
    if args.dataset_type == 'regression_with_binning':
        output_size = args.num_bins * num_tasks
    else:
        output_size = num_tasks
    encoder = MPN(args)
    modules = [
        encoder,
        nn.Linear(args.hidden_size, args.hidden_size),
        nn.ReLU(),
        nn.Linear(args.hidden_size, output_size)
    ]
    if args.dataset_type == 'classification':
        modules.append(nn.Sigmoid())

    model = nn.Sequential(*modules)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace):
        """Initializes the MPN."""
        super(MPN, self).__init__()
        self.hidden_size = args.hidden_size
        self.depth = args.depth
        self.dropout = args.dropout
        self.attention = args.attention
        self.message_attention = args.message_attention
        self.master_node = args.master_node
        self.master_dim = args.master_dim

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.W_i = nn.Linear(get_atom_fdim(args) + get_bond_fdim(args), args.hidden_size, bias=False)
        if self.message_attention:
            self.num_heads = args.message_attention_heads
            self.W_h = nn.Linear(self.num_heads*args.hidden_size, args.hidden_size, bias=False)
        else:
            self.W_h = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.W_o = nn.Linear(get_atom_fdim(args) + args.hidden_size, args.hidden_size)
        if self.attention:
            self.W_a = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
            self.W_b = nn.Linear(args.hidden_size, args.hidden_size)
        if self.message_attention:
            self.W_ma = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size, bias=False)
                                       for _ in range(self.num_heads)])
            # uncomment this later if you want attention over binput + nei_message? or on atom incoming at end
            # self.W_ma2 = nn.Linear(hidden_size, 1, bias=False)
        if self.master_node:
            self.GRU_master = nn.GRU(args.hidden_size, args.master_dim)
            self.W_master_out = nn.Linear(args.master_dim, args.hidden_size)

        if args.activation == "ReLU":
            self.act_func = nn.ReLU()
        elif args.activation == "LeakyReLU":
            self.act_func = nn.LeakyReLU(0.1)
        elif args.activation == "PReLU":
            self.act_func = nn.PReLU()
        elif args.activation == 'tanh':
            self.act_func = nn.Tanh()
        else:
            raise ValueError('Activation "{}" not supported.'.format(args.activation))

    def forward(self, mol_graph: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]) -> torch.Tensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A tuple containing a batch of molecular graphs.
        :return: A PyTorch tensor containing the encoding of the graph.
        """
        fatoms, fbonds, agraph, bgraph, scope = mol_graph
        if next(self.parameters()).is_cuda:
            fatoms, fbonds, agraph, bgraph = fatoms.cuda(), fbonds.cuda(), agraph.cuda(), bgraph.cuda()

        binput = self.W_i(fbonds)
        message = self.act_func(binput)

        for i in range(self.depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            if self.message_attention:
                message = message.unsqueeze(1).repeat((1, nei_message.size(1), 1))  # num_bonds x 1 x hidden
                attention_scores = [(self.W_ma[i](nei_message) * message).sum(dim=2)
                                    for i in range(self.num_heads)]  # num_bonds x maxnb
                attention_weights = [torch.softmax(attention_scores[i], dim=1)
                                     for i in range(self.num_heads)]  # num_bonds x maxnb
                message_components = [nei_message * attention_weights[i].unsqueeze(2).repeat((1, 1, self.hidden_size))
                                      for i in range(self.num_heads)]  # num_bonds x maxnb x hidden
                message_components = [component.sum(dim=1) for component in message_components]  # num_bonds x hidden
                nei_message = torch.cat(message_components, dim=1)  # num_bonds x 3*hidden
            else:
                nei_message = nei_message.sum(dim=1)  # num_bonds x hidden
            nei_message = self.W_h(nei_message)
            if self.master_node:
                # master_state = self.W_master_in(self.act_func(nei_message.sum(dim=0))) #try something like this to preserve invariance for master node
                master_state = self.GRU_master(nei_message.unsqueeze(1))
                master_state = master_state[-1].squeeze(0) #this actually doesn't preserve order invariance anymore
                message = self.act_func(binput + nei_message + self.W_master_out(master_state).repeat((nei_message.size(0), 1)))
            else:
                message = self.act_func(binput + nei_message)
            message = self.dropout_layer(message)  # num_bonds x hidden

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = self.act_func(self.W_o(ainput))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        mol_vecs = []
        for st, le in scope:
            cur_hiddens = atom_hiddens.narrow(0, st, le)

            if self.attention:
                att_w = torch.matmul(self.W_a(cur_hiddens), cur_hiddens.t())
                att_w = F.softmax(att_w, dim=1)
                att_hiddens = torch.matmul(att_w, cur_hiddens)
                att_hiddens = self.act_func(self.W_b(att_hiddens))
                att_hiddens = self.dropout_layer(att_hiddens)
                mol_vec = (cur_hiddens + att_hiddens)
            else:
                mol_vec = cur_hiddens

            mol_vec = mol_vec.sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs
