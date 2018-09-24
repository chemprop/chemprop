from copy import deepcopy
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn.functional as F
import torch.nn as nn

from nn_utils import index_select_ND

ELEM_LIST = list(range(100))
HYBRID_LIST = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
ATOM_FDIM = 100 + len(HYBRID_LIST) + 6 + 5 + 4 + 7 + 5 + 3 + 1
BOND_FDIM = 6 + 6
MAX_NB = 12

# Memoization
SMILES_TO_FEATURES = {}


def get_atom_fdim(three_d: bool = False) -> int:
    """
    Gets the dimensionality of atom features.

    :param three_d: Whether to include 3D coordinates in atom features.
    :return: The number of atom features.
    """
    return ATOM_FDIM + (3 * three_d)


def get_bond_fdim(three_d: bool = False) -> int:
    """
    Gets the dimensionality of bond features.

    :param three_d: Whether to include 3D distance between atoms in bond features.
    :return: The number of bond features.
    """
    return BOND_FDIM + (1 * three_d)


def onek_encoding_unk(x: int, allowable_set: List[int]) -> List[bool]:
    """
    Creates a one-hot encoding.

    :param x: The element of the encoding which should be one.
    :param allowable_set: A list of allowed values.
    :return: A one-hot encoding of x in the allowable_set. If x is not in the allowable set,
    then the final element in the allowable_set is set to 1 while the others are zero.
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom: Chem.rdchem.Atom, atom_position: List[int] = None) -> torch.Tensor:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param atom_position: The 3D coordinates of the atom.
    :return: A PyTorch tensor containing the atom features.
    """
    return torch.Tensor(onek_encoding_unk(atom.GetAtomicNum() - 1, ELEM_LIST)
                        + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                        + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                        + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])
                        + onek_encoding_unk(int(atom.GetImplicitValence()), [0, 1, 2, 3, 4, 5, 6])
                        + onek_encoding_unk(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4])
                        + onek_encoding_unk(int(atom.GetHybridization()), HYBRID_LIST)
                        + onek_encoding_unk(int(atom.GetNumRadicalElectrons()), [0, 1, 2])
                        + [atom.GetIsAromatic()]
                        + (atom_position if atom_position is not None else []))


def bond_features(bond: Chem.rdchem.Bond, distance: float = None) -> torch.Tensor:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :param distance: The distance in 3D space between the two atoms in the bond.
    :return: A PyTorch tensor containing the bond features.
    """
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
             bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])
    fdistance = [distance] if distance is not None else []
    return torch.Tensor(fbond + fstereo + fdistance)


def mol2graph(mol_batch: List[str],
              addHs: bool = False,
              three_d: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Converts a list of SMILES strings to a batch of molecular graphs consisting of of PyTorch tensors.

    :param mol_batch: A list of SMILES strings.
    :param addHs: Whether to include Hydrogen atoms in each molecular graph.
    :param three_d: Whether to include 3D information in atom and bond features.
    :return: A tuple of tensors representing a batch of molecular graphs. TODO: Better explanation.
    """
    padding = torch.zeros(get_atom_fdim(three_d) + get_bond_fdim(three_d))
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
            if addHs:
                mol = Chem.AddHs(mol)
            n_atoms = mol.GetNumAtoms()

            # Get 3D info
            if three_d:
                mol3d = deepcopy(mol)  # make sure not to mess up original molecule when other features are extracted
                mol3d = Chem.AddHs(mol3d)
                AllChem.EmbedMolecule(mol3d, AllChem.ETKDG())
                conformer = mol3d.GetConformer()

            # Get atom features
            for atom in mol.GetAtoms():
                atom_position = list(conformer.GetAtomPosition(atom.GetIdx())) if three_d else None
                mol_fatoms.append(atom_features(atom, atom_position=atom_position))

            # Get bond features
            for bond in mol.GetBonds():
                a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
                a1_idx, a2_idx = a1.GetIdx(), a2.GetIdx()

                if three_d:
                    a1_position = np.array(conformer.GetAtomPosition(a1.GetIdx()))
                    a2_position = np.array(conformer.GetAtomPosition(a2.GetIdx()))
                    distance = np.linalg.norm(a1_position - a2_position)
                else:
                    distance = None

                mol_all_bonds.append((a1_idx, a2_idx))
                mol_fbonds.append(torch.cat([mol_fatoms[a1_idx], bond_features(bond, distance=distance)], 0))

                mol_all_bonds.append((a2_idx, a1_idx))
                mol_fbonds.append(torch.cat([mol_fatoms[a2_idx], bond_features(bond, distance=distance)], 0))

            # Memoize
            SMILES_TO_FEATURES[smiles] = (mol_fatoms, mol_fbonds, mol_all_bonds, n_atoms)

        # Add molecule features to batch features
        fatoms.extend(mol_fatoms)
        fbonds.extend(mol_fbonds)

        in_bonds.extend([[] for _ in range(n_atoms)])

        for a1_idx, a2_idx in mol_all_bonds:
            a1_idx += total_atoms
            a2_idx += total_atoms

            bond_idx = len(all_bonds)
            all_bonds.append((a1_idx, a2_idx))
            in_bonds[a2_idx].append(bond_idx)

        scope.append((total_atoms, n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    agraph = torch.zeros(total_atoms, MAX_NB).long()
    bgraph = torch.zeros(total_bonds, MAX_NB).long()

    for a in range(total_atoms):
        for i, b in enumerate(in_bonds[a]):
            agraph[a, i] = b

    for b1 in range(1, total_bonds):
        x, y = all_bonds[b1]
        for i, b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1, i] = b2

    return fatoms, fbonds, agraph, bgraph, scope


def build_MPN(hidden_size: int,
              depth: int,
              num_tasks: int,
              sigmoid: bool,
              dropout: float = 0.0,
              activation: str = "ReLU",
              attention: bool = False,
              three_d: bool = False) -> nn.Module:
    """
    Builds a message passing neural network including final linear layers and initializes parameters.

    :param hidden_size: Dimensionality of hidden layers.
    :param depth: Number of message passing steps.
    :param num_tasks: Number of tasks to predict.
    :param sigmoid: Whether to add a sigmoid layer at the end for classification.
    :param dropout: Dropout probability.
    :param activation: Activation function.
    :param attention: Whether to perform self attention over the atoms in a molecule.
    :param three_d: Whether to include 3D information in atom and bond features.
    :return: An nn.Module containing the MPN encoder along with final linear layers with parameters initialized.
    """
    encoder = MPN(
        hidden_size=hidden_size,
        depth=depth,
        dropout=dropout,
        activation=activation,
        attention=attention,
        three_d=three_d
    )
    modules = [
        encoder,
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_tasks)
    ]
    if sigmoid:
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

    def __init__(self,
                 hidden_size: int,
                 depth: int,
                 dropout: float = 0.0,
                 activation: str = "ReLU",
                 attention: bool = False,
                 three_d: bool = False):
        """
        Initializes the MPN.

        :param hidden_size: Hidden dimensionality.
        :param depth: Number of message passing steps.
        :param dropout: Dropout probability.
        :param activation: Activation function.
        :param attention: Whether to perform self attention over the atoms in a molecule.
        :param three_d: Whether to include 3D information in atom and bond features.
        """
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.attention = attention

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.W_i = nn.Linear(get_atom_fdim(three_d) + get_bond_fdim(three_d), hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(get_atom_fdim(three_d) + hidden_size, hidden_size)
        if self.attention:
            self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_b = nn.Linear(hidden_size, hidden_size)

        if activation == "ReLU":
            self.act_func = nn.ReLU()
        elif activation == "LeakyReLU":
            self.act_func = nn.LeakyReLU(0.1)
        elif activation == "PReLU":
            self.act_func = nn.PReLU()
        elif activation == 'tanh':
            self.act_func = nn.Tanh()
        else:
            raise ValueError('Activation "{}" not supported.'.format(activation))

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
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = self.act_func(binput + nei_message)
            message = self.dropout_layer(message)

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
