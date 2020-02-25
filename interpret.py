import sys, os
import argparse
import math
import numpy as np
import tqdm
from rdkit import Chem
from functools import partial

from chemprop.train import predict
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers

MIN_ATOMS = 15
C_PUCT = 10

class ChempropModel():
    
    def __init__(self, checkpoint_dir):
        self.checkpoints = []
        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    fname = os.path.join(root, fname)
                    self.scaler, self.features_scaler = load_scalers(fname)
                    self.train_args = load_args(fname)
                    model = load_checkpoint(fname, cuda=True)
                    self.checkpoints.append(model)

    def __call__(self, smiles, batch_size=500):
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False, args=self.train_args)
        valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
        full_data = test_data
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])

        if self.train_args.features_scaling:
            test_data.normalize_features(self.features_scaler)

        sum_preds = [] 
        for model in self.checkpoints:
            model_preds = predict(
                model=model,
                data=test_data,
                batch_size=batch_size,
                scaler=self.scaler,
                disable_progress_bar=True
            )
            sum_preds.append(np.array(model_preds))

        # Ensemble predictions
        sum_preds = sum(sum_preds)
        avg_preds = sum_preds / len(self.checkpoints)
        return avg_preds


class MCTSNode():

    def __init__(self, smiles, atoms, W=0, N=0, P=0):
        self.smiles = smiles
        self.atoms = set(atoms)
        self.children = []
        self.W = W
        self.N = N
        self.P = P

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return C_PUCT * self.P * math.sqrt(n) / (1 + self.N)


def find_clusters(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: #special case
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( (a1,a2) )

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for i in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls


def __extract_subgraph(mol, selected_atoms):
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [bond for bond in aroma_bonds if bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots


def extract_subgraph(smiles, selected_atoms): 
    # try with kekulization
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    subgraph, roots = __extract_subgraph(mol, selected_atoms) 
    subgraph = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
    subgraph = Chem.MolFromSmiles(subgraph)

    mol = Chem.MolFromSmiles(smiles)  # de-kekulize
    if subgraph is not None and mol.HasSubstructMatch(subgraph):
        return Chem.MolToSmiles(subgraph), roots

    # If fails, try without kekulization
    subgraph, roots = __extract_subgraph(mol, selected_atoms) 
    subgraph = Chem.MolToSmiles(subgraph)
    subgraph = Chem.MolFromSmiles(subgraph)
    if subgraph is not None:
        return Chem.MolToSmiles(subgraph), roots
    else:
        return None, None


def mcts_rollout(node, state_map, orig_smiles, clusters, atom_cls, nei_cls, scoring_function):
    cur_atoms = node.atoms
    if len(cur_atoms) <= MIN_ATOMS:
        return node.P

    # Expand if this node has never been visited
    if len(node.children) == 0:
        cur_cls = set( [i for i,x in enumerate(clusters) if x <= cur_atoms] )
        for i in cur_cls:
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]
            if len(nei_cls[i] & cur_cls) == 1 or len(clusters[i]) == 2 and len(leaf_atoms) == 1:
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)
                if new_smiles in state_map:
                    new_node = state_map[new_smiles] # merge identical states
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0: return node.P  # cannot find leaves

        scores = scoring_function([x.smiles for x in node.children])
        for child, score in zip(node.children, scores):
            child.P = score
        
    sum_count = sum([c.N for c in node.children])
    selected_node = max(node.children, key=lambda x : x.Q() + x.U(sum_count))
    v = mcts_rollout(selected_node, state_map, orig_smiles, clusters, atom_cls, nei_cls, scoring_function)
    selected_node.W += v
    selected_node.N += 1
    return v


def mcts(smiles, scoring_function, n_rollout, max_atoms, prop_delta): 
    mol = Chem.MolFromSmiles(smiles)
    if mol.GetNumAtoms() > 50:
        n_rollout = 1

    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    for i,cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - set([i])
        clusters[i] = set(list(cls))
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])
    
    root = MCTSNode( smiles, set(range(mol.GetNumAtoms())) ) 
    state_map = {smiles : root}
    for _ in range(n_rollout):
        mcts_rollout(root, state_map, smiles, clusters, atom_cls, nei_cls, scoring_function)

    rationales = [node for _,node in state_map.items() if len(node.atoms) <= max_atoms and node.P >= prop_delta]
    return rationales


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--rollout', type=int, default=20)
    parser.add_argument('--c_puct', type=float, default=10)
    parser.add_argument('--max_atoms', type=int, default=20)
    parser.add_argument('--min_atoms', type=int, default=8)
    parser.add_argument('--prop_delta', type=float, default=0.5)
    parser.add_argument('--property_id', type=int, default=1)
    args = parser.parse_args()

    chemprop_model = ChempropModel(args.checkpoint_dir)
    scoring_function = lambda x:chemprop_model(x)[:, args.property_id - 1]

    C_PUCT = args.c_puct
    MIN_ATOMS = args.min_atoms

    with open(args.data_path) as f:
        header = next(f)
        data = [line.split(',')[0] for line in f]

    header = header.split(',')
    if len(header) > args.property_id:
        print('smiles,%s,rationale,rationale_score' % (header[args.property_id],))
    else:
        print('smiles,score,rationale,rationale_score')

    for smiles in data:
        smiles = smiles.strip("\r\n ")
        score = scoring_function([smiles])[0]
        if score > args.prop_delta:
            rationales = mcts(smiles, scoring_function=scoring_function, 
                                  n_rollout=args.rollout, 
                                  max_atoms=args.max_atoms, 
                                  prop_delta=args.prop_delta)
        else:
            rationales = []

        if len(rationales) == 0:
            print("%s,%.3f,," % (smiles, score))
        else:
            min_size = min([len(x.atoms) for x in rationales])
            min_rationales = [x for x in rationales if len(x.atoms) == min_size]
            rats = sorted(min_rationales, key=lambda x:x.P, reverse=True)
            print("%s,%.3f,%s,%.3f" % (smiles, score, rats[0].smiles, rats[0].P))

