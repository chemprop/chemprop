import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F

from nnutils import *

ELEM_LIST = range(100)
HYBRID_LIST = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
ATOM_FDIM = 100 + len(HYBRID_LIST) + 6 + 5 + 4 + 7 + 5 + 3 + 1
BOND_FDIM = 6 + 6
MAX_NB = 12

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetAtomicNum() - 1, ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + onek_encoding_unk(int(atom.GetImplicitValence()), [0,1,2,3,4,5,6])
            + onek_encoding_unk(int(atom.GetTotalNumHs()), [0,1,2,3,4])
            + onek_encoding_unk(int(atom.GetHybridization()), HYBRID_LIST)
            + onek_encoding_unk(int(atom.GetNumRadicalElectrons()), [0,1,2])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

def mol2graph(mol_batch, addHs=False):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms,fbonds = [],[padding] #Ensure bond is 1-indexed
    in_bonds,all_bonds = [],[(-1,-1)] #Ensure bond is 1-indexed
    scope = []
    total_atoms = 0

    for smiles in mol_batch:
        mol = Chem.MolFromSmiles(smiles)
        if addHs:
            mol = Chem.AddHs(mol)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append( atom_features(atom) )
            in_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() + total_atoms
            y = a2.GetIdx() + total_atoms

            b = len(all_bonds) 
            all_bonds.append((x,y))
            fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y,x))
            fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
            in_bonds[x].append(b)
        
        scope.append((total_atoms,n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    agraph = torch.zeros(total_atoms,MAX_NB).long()
    bgraph = torch.zeros(total_bonds,MAX_NB).long()

    for a in xrange(total_atoms):
        for i,b in enumerate(in_bonds[a]):
            agraph[a,i] = b

    for b1 in xrange(1, total_bonds):
        x,y = all_bonds[b1]
        for i,b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1,i] = b2

    return fatoms, fbonds, agraph, bgraph, scope

class MPN(nn.Module):

    def __init__(self, hidden_size, depth, dropout=0, act_func="ReLU"):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)
        if act_func == "ReLU":
            self.act_func = nn.ReLU()
        elif act_func == "LeakyReLU":
            self.act_func = nn.LeakyReLU(0.1)
        elif act_func == "PReLU":
            self.act_func = nn.PReLU()

    def forward(self, mol_graph):
        fatoms,fbonds,agraph,bgraph,scope = mol_graph
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)

        binput = self.W_i(fbonds)
        message = self.act_func(binput)

        for i in xrange(self.depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = self.act_func(binput + nei_message)
            if self.dropout > 0:
                message = F.dropout(message, self.dropout, self.training)

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = self.act_func(self.W_o(ainput))
        if self.dropout > 0:
            atom_hiddens = F.dropout(atom_hiddens, self.dropout, self.training)
        
        mol_vecs = []
        for st,le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

