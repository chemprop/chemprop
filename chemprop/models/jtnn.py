from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple

import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import torch
import torch.nn as nn

from .mpn import MPNEncoder
from chemprop.nn_utils import GraphGRU, index_select_ND

MAX_NB = 20
MST_MAX_WEIGHT = 100

# Memoize
SMILES_TO_MOLTREE = dict()


class Vocab:
    def __init__(self, smiles_list: List[str]):
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}

    def get_index(self, smiles: str) -> int:
        return self.vmap[smiles]

    def get_smiles(self, idx: int) -> str:
        return self.vocab[idx]

    def size(self) -> int:
        return len(self.vocab)


class MolTreeNode:
    def __init__(self, smiles: str, clique: List[int] = None):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)

        if clique is None:
            clique = []
        self.clique = deepcopy(clique)
        self.neighbors = []

    def add_neighbor(self, nei_node: int):
        self.neighbors.append(nei_node)


class MolTree:
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        cliques, edges = tree_decomp(self.mol)
        self.nodes = []
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0:
                root = i

        for x, y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])

        if root > 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

    def size(self) -> int:
        return len(self.nodes)


def get_mol(smiles: str) -> Chem.rdchem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)

    return mol


def get_smiles(mol: Chem.rdchem.Mol) -> str:
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception:
        return None

    return mol


def copy_atom(atom: Chem.rdchem.Atom) -> Chem.rdchem.Atom:
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())

    return new_atom


def copy_edit_mol(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)

    return new_mol


def get_clique_mol(mol: Chem.rdchem.Mol, atoms: List[int]) -> Chem.rdchem.Mol:
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None

    return new_mol


def tree_decomp(mol: Chem.rdchem.Mol) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    n_atoms = mol.GetNumAtoms()
    cliques = []
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 0:
            cliques.append([atom.GetIdx()])

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):  # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1
        elif len(rings) > 2:  # Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(inter)  # cnei[i] < cnei[j] by construction

    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    # Compute Maximum Spanning Tree
    row, col, data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]

    return cliques, edges


class JTNN(nn.Module):
    def __init__(self, args: Namespace):
        super(JTNN, self).__init__()

        with open(args.vocab_path) as f:
            self.vocab = Vocab([line.strip("\r\n ") for line in f])

        self.hidden_size = args.hidden_size
        self.depth = args.depth
        self.args = args

        self.jtnn = MPNEncoder(args, atom_fdim=self.hidden_size, bond_fdim=self.hidden_size)
        self.embedding = nn.Embedding(self.vocab.size(), self.hidden_size)
        self.mpn = MPN(args)

    def forward(self, smiles_batch: List[str]):
        # Get MolTrees with memoization
        mol_batch = [SMILES_TO_MOLTREE[smiles]
                     if smiles in SMILES_TO_MOLTREE else SMILES_TO_MOLTREE.setdefault(smiles, MolTree(smiles))
                     for smiles in smiles_batch]
        fnode, fmess, node_graph, mess_graph, scope = self.tensorize(mol_batch)

        if next(self.parameters()).is_cuda:
            fnode, fmess, node_graph, mess_graph = fnode.cuda(), fmess.cuda(), node_graph.cuda(), mess_graph.cuda()

        fnode = self.embedding(fnode)
        fmess = index_select_ND(fnode, fmess)
        tree_vec = self.jtnn((fnode, fmess, node_graph, mess_graph, scope, []))
        mol_vec = self.mpn(smiles_batch)

        return torch.cat([tree_vec, mol_vec], dim=-1)

    def tensorize(self, tree_batch: List[MolTree]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        node_batch = []
        scope = []
        for tree in tree_batch:
            scope.append((len(node_batch), len(tree.nodes)))
            for i, node in enumerate(tree.nodes):
                node.idx = len(node_batch) + i
            node_batch.extend(tree.nodes)

        messages, mess_dict = [None], {}
        fnode = []
        for x in node_batch:
            wid = self.vocab.get_index(x.smiles)
            fnode.append(wid)
            for y in x.neighbors:
                mess_dict[(x.idx, y.idx)] = len(messages)
                messages.append((x, y))
                mess_dict[(y.idx, x.idx)] = len(messages)
                messages.append((y, x))

        node_graph = [[] for _ in range(len(node_batch))]
        mess_graph = [[] for _ in range(len(messages))]
        fmess = [0] * len(messages)

        for x, y in messages[1:]:
            mid1 = mess_dict[(x.idx, y.idx)]
            fmess[mid1] = x.idx
            node_graph[y.idx].append(mid1)
            for z in y.neighbors:
                if z.idx == x.idx: continue
                mid2 = mess_dict[(y.idx, z.idx)]
                mess_graph[mid2].append(mid1)

        max_len = max([len(t) for t in node_graph])
        for t in node_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        max_len = max([len(t) for t in mess_graph])
        for t in mess_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        mess_graph = torch.LongTensor(mess_graph)
        node_graph = torch.LongTensor(node_graph)
        fmess = torch.LongTensor(fmess)
        fnode = torch.LongTensor(fnode)

        return fnode, fmess, node_graph, mess_graph, scope


class JTNNEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, depth: int):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.GRU = GraphGRU(hidden_size, hidden_size, depth)
        self.outputNN = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )

    def forward(self,
                fnode: torch.Tensor,
                fmess: torch.Tensor,
                node_graph: torch.Tensor,
                mess_graph: torch.Tensor,
                scope: List[Tuple[int, int]]) -> torch.Tensor:
        messages = torch.zeros(mess_graph.size(0), self.hidden_size)

        if next(self.parameters()).is_cuda:
            fnode, fmess, node_graph, mess_graph, messages = fnode.cuda(), fmess.cuda(), node_graph.cuda(), mess_graph.cuda(), messages.cuda()

        fnode = self.embedding(fnode)
        fmess = index_select_ND(fnode, fmess)
        messages = self.GRU(messages, fmess, mess_graph)

        mess_nei = index_select_ND(messages, node_graph)
        fnode = torch.cat([fnode, mess_nei.sum(dim=1)], dim=-1)
        fnode = self.outputNN(fnode)
        tree_vec = []
        for st, le in scope:
            tree_vec.append(fnode.narrow(0, st, le).mean(dim=0))

        return torch.stack(tree_vec, dim=0)


"""
class JTNNEncoder(nn.Module):

    def __init__(self, vocab, hidden_size, embedding=None):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, root_batch):
        orders = []
        for root in root_batch:
            order = get_prop_order(root)
            orders.append(order)
        
        h = {}
        max_depth = max([len(x) for x in orders])
        padding = create_var(torch.zeros(self.hidden_size), False)

        for t in range(max_depth):
            prop_list = []
            for order in orders:
                if t < len(order):
                    prop_list.extend(order[t])

            cur_x = []
            cur_h_nei = []
            for node_x,node_y in prop_list:
                x,y = node_x.idx,node_y.idx
                cur_x.append(node_x.wid)

                h_nei = []
                for node_z in node_x.neighbors:
                    z = node_z.idx
                    if z == y: continue
                    h_nei.append(h[(z,x)])

                pad_len = MAX_NB - len(h_nei)
                if pad_len < 0: 
                    h_nei = h_nei[:MAX_NB]
                else:
                    h_nei.extend([padding] * pad_len)
                cur_h_nei.extend(h_nei)

            cur_x = create_var(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x)
            cur_h_nei = torch.cat(cur_h_nei, dim=0).view(-1,MAX_NB,self.hidden_size)

            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
            for i,m in enumerate(prop_list):
                x,y = m[0].idx,m[1].idx
                h[(x,y)] = new_h[i]

        root_vecs = node_aggregate(root_batch, h, self.embedding, self.W)

        return h, root_vecs

def get_prop_order(root):
    queue = deque([root])
    visited = set([root.idx])
    root.depth = 0
    order1,order2 = [],[]
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.neighbors:
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth-1].append( (x,y) )
                order2[y.depth-1].append( (y,x) )
    order = order2[::-1] + order1
    return order

def node_aggregate(nodes, h, embedding, W):
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_dim
    padding = create_var(torch.zeros(hidden_size), False)

    for node_x in nodes:
        x_idx.append(node_x.wid)
        nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
        pad_len = MAX_NB - len(nei)
        if pad_len < 0:
            nei = nei[:MAX_NB]
        else:
            nei.extend([padding] * pad_len)
        h_nei.extend(nei)
    
    h_nei = torch.cat(h_nei, dim=0).view(-1,MAX_NB,hidden_size)
    sum_h_nei = h_nei.sum(dim=1)
    x_vec = create_var(torch.LongTensor(x_idx))
    x_vec = embedding(x_vec)
    node_vec = torch.cat([x_vec, sum_h_nei], dim=1)
    return nn.ReLU()(W(node_vec))
"""
