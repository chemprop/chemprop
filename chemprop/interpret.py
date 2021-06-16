import math
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from rdkit import Chem

from chemprop.args import InterpretArgs
from chemprop.data import get_data_from_smiles, get_header, get_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.train import predict
from chemprop.utils import load_args, load_checkpoint, load_scalers, timeit


MIN_ATOMS = 15
C_PUCT = 10


class ChempropModel:
    """A :class:`ChempropModel` is a wrapper around a :class:`~chemprop.models.model.MoleculeModel` for interpretation."""

    def __init__(self, args: InterpretArgs) -> None:
        """
        :param args: A :class:`~chemprop.args.InterpretArgs` object containing arguments for interpretation.
        """
        self.args = args
        self.train_args = load_args(args.checkpoint_paths[0])

        # If features were used during training, they must be used when predicting
        if ((self.train_args.features_path is not None or self.train_args.features_generator is not None)
                and args.features_generator is None):
            raise ValueError('Features were used during training so they must be specified again during prediction '
                             'using the same type of features as before (with --features_generator <generator> '
                             'and using --no_features_scaling if applicable).')

        if self.train_args.atom_descriptors_size > 0 or self.train_args.atom_features_size > 0 or self.train_args.bond_features_size > 0:
            raise NotImplementedError('The interpret function does not yet work with additional atom or bond features')

        self.scaler, self.features_scaler, self.atom_descriptor_scaler, self.bond_feature_scaler = load_scalers(args.checkpoint_paths[0])
        self.checkpoints = [load_checkpoint(checkpoint_path, device=args.device) for checkpoint_path in args.checkpoint_paths]

    def __call__(self, smiles: List[str], batch_size: int = 500) -> List[List[float]]:
        """
        Makes predictions on a list of SMILES.

        :param smiles: A list of SMILES to make predictions on.
        :param batch_size: The batch size.
        :return: A list of lists of floats containing the predicted values.
        """
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False, features_generator=self.args.features_generator)
        valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])

        if self.train_args.features_scaling:
            test_data.normalize_features(self.features_scaler)
        if self.train_args.atom_descriptor_scaling and self.args.atom_descriptors is not None:
            test_data.normalize_features(self.atom_descriptor_scaler, scale_atom_descriptors=True)
        if self.train_args.bond_feature_scaling and self.args.bond_features_size > 0:
            test_data.normalize_features(self.bond_feature_scaler, scale_bond_features=True)

        test_data_loader = MoleculeDataLoader(dataset=test_data, batch_size=batch_size)

        sum_preds = []
        for model in self.checkpoints:
            model_preds = predict(
                model=model,
                data_loader=test_data_loader,
                scaler=self.scaler,
                disable_progress_bar=True
            )
            sum_preds.append(np.array(model_preds))

        # Ensemble predictions
        sum_preds = sum(sum_preds)
        avg_preds = sum_preds / len(self.checkpoints)

        return avg_preds


class MCTSNode:
    """A :class:`MCTSNode` represents a node in a Monte Carlo Tree Search."""

    def __init__(self, smiles: str, atoms: List[int], W: float = 0, N: int = 0, P: float = 0) -> None:
        """
        :param smiles: The SMILES for the substructure at this node.
        :param atoms: A list of atom indices represented by this node.
        :param W: The W value of this node.
        :param N: The N value of this node.
        :param P: The P value of this node.
        """
        self.smiles = smiles
        self.atoms = set(atoms)
        self.children = []
        self.W = W
        self.N = N
        self.P = P

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0

    def U(self, n: int) -> float:
        return C_PUCT * self.P * math.sqrt(n) / (1 + self.N)


def find_clusters(mol: Chem.Mol) -> Tuple[List[Tuple[int, ...]], List[List[int]]]:
    """
    Finds clusters within the molecule.

    :param mol: An RDKit molecule.
    :return: A tuple containing a list of atom tuples representing the clusters
             and a list of lists of atoms in each cluster.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:  # special case
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append((a1, a2))

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for _ in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls


def __extract_subgraph(mol: Chem.Mol, selected_atoms: Set[int]) -> Tuple[Chem.Mol, List[int]]:
    """
    Extracts a subgraph from an RDKit molecule given a set of atom indices.

    :param mol: An RDKit molecule from which to extract a subgraph.
    :param selected_atoms: The atoms which form the subgraph to be extracted.
    :return: A tuple containing an RDKit molecule representing the subgraph
             and a list of root atom indices from the selected indices.
    """
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
        aroma_bonds = [bond for bond in aroma_bonds if
                       bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots


def extract_subgraph(smiles: str, selected_atoms: Set[int]) -> Tuple[str, List[int]]:
    """
    Extracts a subgraph from a SMILES given a set of atom indices.

    :param smiles: A SMILES from which to extract a subgraph.
    :param selected_atoms: The atoms which form the subgraph to be extracted.
    :return: A tuple containing a SMILES representing the subgraph
             and a list of root atom indices from the selected indices.
    """
    # try with kekulization
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    subgraph, roots = __extract_subgraph(mol, selected_atoms)
    try:
        subgraph = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
        subgraph = Chem.MolFromSmiles(subgraph)
    except Exception:
        subgraph = None

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


def mcts_rollout(node: MCTSNode,
                 state_map: Dict[str, MCTSNode],
                 orig_smiles: str,
                 clusters: List[Set[int]],
                 atom_cls: List[Set[int]],
                 nei_cls: List[Set[int]],
                 scoring_function: Callable[[List[str]], List[float]]) -> float:
    """
    A Monte Carlo Tree Search rollout from a given :class:`MCTSNode`.

    :param node: The :class:`MCTSNode` from which to begin the rollout.
    :param state_map: A mapping from SMILES to :class:`MCTSNode`.
    :param orig_smiles: The original SMILES of the molecule.
    :param clusters: Clusters of atoms.
    :param atom_cls: Atom indices in the clusters.
    :param nei_cls: Neighboring clusters.
    :param scoring_function: A function for scoring subgraph SMILES using a Chemprop model.
    :return: The score of this MCTS rollout.
    """
    cur_atoms = node.atoms
    if len(cur_atoms) <= MIN_ATOMS:
        return node.P

    # Expand if this node has never been visited
    if len(node.children) == 0:
        cur_cls = set([i for i, x in enumerate(clusters) if x <= cur_atoms])
        for i in cur_cls:
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]
            if len(nei_cls[i] & cur_cls) == 1 or len(clusters[i]) == 2 and len(leaf_atoms) == 1:
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)
                if new_smiles in state_map:
                    new_node = state_map[new_smiles]  # merge identical states
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0:
            return node.P  # cannot find leaves

        scores = scoring_function([[x.smiles] for x in node.children])
        for child, score in zip(node.children, scores):
            child.P = score

    sum_count = sum(c.N for c in node.children)
    selected_node = max(node.children, key=lambda x: x.Q() + x.U(sum_count))
    v = mcts_rollout(selected_node, state_map, orig_smiles, clusters, atom_cls, nei_cls, scoring_function)
    selected_node.W += v
    selected_node.N += 1

    return v


def mcts(smiles: str,
         scoring_function: Callable[[List[str]], List[float]],
         n_rollout: int,
         max_atoms: int,
         prop_delta: float) -> List[MCTSNode]:
    """
    Runs the Monte Carlo Tree Search algorithm.

    :param smiles: The SMILES of the molecule to perform the search on.
    :param scoring_function: A function for scoring subgraph SMILES using a Chemprop model.
    :param n_rollout: THe number of MCTS rollouts to perform.
    :param max_atoms: The maximum number of atoms allowed in an extracted rationale.
    :param prop_delta: The minimum required property value for a satisfactory rationale.
    :return: A list of rationales each represented by a :class:`MCTSNode`.
    """
            
    mol = Chem.MolFromSmiles(smiles)
    if mol.GetNumAtoms() > 50:
        n_rollout = 1

    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    for i, cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - {i}
        clusters[i] = set(list(cls))
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])

    root = MCTSNode(smiles, set(range(mol.GetNumAtoms())))
    state_map = {smiles: root}
    for _ in range(n_rollout):
        mcts_rollout(root, state_map, smiles, clusters, atom_cls, nei_cls, scoring_function)

    rationales = [node for _, node in state_map.items() if len(node.atoms) <= max_atoms and node.P >= prop_delta]

    return rationales

@timeit()
def interpret(args: InterpretArgs) -> None:
    """
    Runs interpretation of a Chemprop model using the Monte Carlo Tree Search algorithm.

    :param args: A :class:`~chemprop.args.InterpretArgs` object containing arguments for interpretation.
    """

    if args.number_of_molecules != 1:
        raise ValueError("Interpreting is currently only available for single-molecule models.")
    
    global C_PUCT, MIN_ATOMS

    chemprop_model = ChempropModel(args)

    def scoring_function(smiles: List[str]) -> List[float]:
        return chemprop_model(smiles)[:, args.property_id - 1]

    C_PUCT = args.c_puct
    MIN_ATOMS = args.min_atoms

    all_smiles = get_smiles(path=args.data_path, smiles_columns=args.smiles_columns)
    header = get_header(path=args.data_path)

    property_name = header[args.property_id] if len(header) > args.property_id else 'score'
    print(f'smiles,{property_name},rationale,rationale_score')

    for smiles in all_smiles:
        score = scoring_function([smiles])[0]
        if score > args.prop_delta:
            rationales = mcts(
                smiles=smiles[0],
                scoring_function=scoring_function,
                n_rollout=args.rollout,
                max_atoms=args.max_atoms,
                prop_delta=args.prop_delta
            )
        else:
            rationales = []

        if len(rationales) == 0:
            print(f'{smiles},{score:.3f},,')
        else:
            min_size = min(len(x.atoms) for x in rationales)
            min_rationales = [x for x in rationales if len(x.atoms) == min_size]
            rats = sorted(min_rationales, key=lambda x: x.P, reverse=True)
            print(f'{smiles},{score:.3f},{rats[0].smiles},{rats[0].P:.3f}')


def chemprop_interpret() -> None:
    """Runs interpretation of a Chemprop model.

    This is the entry point for the command line command :code:`chemprop_interpret`.
    """
    interpret(args=InterpretArgs().parse_args())
