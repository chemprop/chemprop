from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from typing import List, Sequence, Tuple

from rdkit.Chem.rdchem import Atom, Bond, HybridizationType
import torch
import numpy as np

from chemprop.rdkit import make_mol


def safe_index(x, xs: Sequence):
    """return both the index of `x` in `xs` (if it exists, else -1) and the total length of `xs`"""
    return xs.index(x) if x in xs else len(xs), len(xs)


@dataclass
class MolGraph:
    """A `MolGraph` represents the graph structure and featurization of a single molecule.

    Attributes
    ----------
    n_atoms : int
        the number of atoms in the molecule
    n_bonds : int
        the number of bonds in the molecule
    X_v : np.ndarray
        the atom features of the molecule
    X_e : np.ndarray
        the bond features of the molecule
    a2b : list[tuple[int]]
        A mapping from an atom index to a list of incoming bond indices.
    b2a : list[int]
        A mapping from a bond index to the index of the atom the bond originates from.
    b2revb
        A mapping from a bond index to the index of the reverse bond.
    """
    n_atoms: int
    n_bonds: int
    X_v: np.ndarray
    X_e: np.ndarray
    a2b: list[tuple[int]]
    b2a: list[int]
    b2revb: list[int]
    b2b: list[int]
    a2a: list[int]


@dataclass
class AtomFeaturizationParams:
    max_atomic_num: InitVar[int] = 100
    atomic_num: list[int] = field(init=False)
    degree: list[int] = field(default_factory=lambda: list(range(6)))
    formal_charge: list[int] = field(default_factory=lambda: [-1, -2, 1, 2, 0])
    chiral_tag: list[int] = field(default_factory=lambda: list(range(4)))
    num_Hs: list[int] = field(default_factory=lambda: list(range(5)))
    hybridization: list[HybridizationType] = field(
        default_factory=lambda: [
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2
        ]
    )

    def __post_init__(self, max_atomic_num: int):
        self.atomic_num = list(range(max_atomic_num))

    def __len__(self):
        """the dimension of an atom feature vector, adding 1 to each set of features for uncommon
        values and 2 at the end to account for aromaticity and mass"""
        return sum(len(getattr(self, field.name)) + 1 for field in fields(self)) + 2


@dataclass
class MolGraphFeaturizer:
    max_atomic_num: int = 100
    atom_feature_params: AtomFeaturizationParams = field(init=False)
    atom_fdim: int = field(init=False)
    extra_atom_fdim: int = 0
    bond_fdim: int = 14
    extra_bond_fdim: int = 0
    atom_messages: bool = False
    keep_h: bool = False
    add_h: bool = False

    def __post_init__(self):
        self.atom_feature_params = AtomFeaturizationParams(self.max_atomic_num)
        self.atom_fdim = len(self.atom_feature_params)
        if self.atom_messages:
            self.bond_fdim += self.atom_fdim

    def featurize(
        self,
        smi: str,
        atom_features_extra: np.ndarray = None,
        bond_features_extra: np.ndarray = None,
    ) -> MolGraph:
        mol = make_mol(smi, self.keep_h, self.add_h)

        n_atoms = mol.GetNumAtoms() 
        n_bonds = mol.GetNumBonds()
        X_v = []
        X_e = []
        a2b = [[]] * n_atoms
        b2a = []
        b2revb = []

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )
            
        X_v = np.stack([self.atom_features(a) for a in mol.GetAtoms()])
        X_e = np.empty((2 * n_bonds, self.bond_fdim))

        if atom_features_extra is not None:
            X_v = np.hstack((X_v, atom_features_extra))

        i = 0
        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                b = mol.GetBondBetweenAtoms(a1, a2)

                if b is None:
                    continue

                x_e = self.bond_features(b)
                if bond_features_extra is not None:
                    x_e += bond_features_extra[b.GetIdx()].tolist()

                b12 = i
                b21 = b12 + 1

                X_e[b12] = np.concatenate((X_v[a1], x_e))
                X_e[b21] = np.concatenate((X_v[a2], x_e))

                a2b[a2].append(b12)
                b2a.append(a1)
                a2b[a1].append(b21)
                b2a.append(a2)
                b2revb.append(b21)
                b2revb.append(b12)

                i += 2

        return MolGraph(n_atoms, 2*n_bonds, X_v, X_e, a2b, b2a, b2revb, None, None)

    def atom_features(self, a: Atom) -> np.ndarray:
        x = np.zeros(self.atom_fdim)

        if a is None:
            return x

        bits_offsets = [
            safe_index((a.GetAtomicNum() -1), self.atom_feature_params.atomic_num),
            safe_index(a.GetTotalDegree(), self.atom_feature_params.degree),
            safe_index(a.GetFormalCharge(), self.atom_feature_params.formal_charge),
            safe_index(int(a.GetChiralTag()), self.atom_feature_params.chiral_tag),
            safe_index(int(a.GetTotalNumHs()), self.atom_feature_params.num_Hs),
            safe_index(int(a.GetHybridization()), self.atom_feature_params.hybridization),
        ]

        i = 0
        for bit, offset in bits_offsets:
            x[i + bit] = 1
            i += offset
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = [0.01 * a.Getmass()]
        
        return x

    def bond_features(self, b: Bond) -> np.ndarray:
        x = np.zeros(self.bond_fdim)
        
        if b is None:
            x[0] = 1
            return x

        bond_type = b.GetBondType()

        if bond_type is not None:
            bt_int = int(bond_type)
            CONJ_BIT = 5
            RING_BIT = 6

            if bt_int in {1, 2, 3, 12}:
                x[max(4, bt_int)] = 1
            if b.GetIsConjugated():
                x[CONJ_BIT] = 1
            if b.IsInRing():
                x[RING_BIT] = 1

        stereo_bit = int(b.GetStereo())
        x[stereo_bit] = 1

        return x


class BatchMolGraph:
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.overwrite_default_atom_features = mol_graphs[0].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs[0].overwrite_default_bond_features
        self.is_reaction = mol_graphs[0].is_reaction
        self.atom_fdim = get_atom_fdim(overwrite_default_atom=self.overwrite_default_atom_features,
                                       is_reaction=self.is_reaction)
        self.bond_fdim = get_bond_fdim(overwrite_default_bond=self.overwrite_default_bond_features,
                                      overwrite_default_atom=self.overwrite_default_atom_features,
                                      is_reaction=self.is_reaction)

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
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

        # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the :class:`BatchMolGraph`.

        The returned components are, in order:

        * :code:`f_atoms`
        * :code:`f_bonds`
        * :code:`a2b`
        * :code:`b2a`
        * :code:`b2revb`
        * :code:`a_scope`
        * :code:`b_scope`

        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                              vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        if atom_messages:
            j = get_bond_fdim(
                atom_messages,
                self.overwrite_default_bond_features,
                self.overwrite_default_atom_features
            )
            f_bonds = self.f_bonds[:,-j:]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each atom index to all the neighboring atom indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a
