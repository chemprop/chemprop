from copy import deepcopy
from dataclasses import InitVar, dataclass, field

import numpy as np
from collections import Counter
from rdkit import Chem

from chemprop.data.molgraph import MolGraph
from chemprop.types import Polymer
from chemprop.featurizers.base import GraphFeaturizer
from chemprop.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin


@dataclass
class PolymerMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Polymer]):
    """A :class:`PolymerMolGraphFeaturizer` is the default implementation of a
    :class:`PolymerMolGraphFeaturizer`

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    degree_of_polym: int, default=1
        the degree of polymerization of the polymer
    polymer_info: list
        the list of polymer information from the input string
        
    References
    ----------
    .. [1] A graph representation of molecular ensembles for polymer property prediction (Chem. Sci. 2022,13, 10486-10498)
           https://pubs.rsc.org/en/content/articlelanding/2022/SC/D2SC02839E
    """
    extra_atom_fdim: InitVar[int] = 0
    extra_bond_fdim: InitVar[int] = 0
    degree_of_poly: InitVar[int] = 1
    polymer_info: list = field(default_factory=list)
       
    
    def tag_atoms_in_repeating_unit(self, mol):
        """
        Tag atoms that are part of the core units, as well as atoms serving to identify attachment points. In addition,
        create a map of bond types based on what bonds are connected to R groups in the input.
        """
        atoms = [a for a in mol.GetAtoms()]
        neighbor_map = {} # Map the R group index of the atom its attached to
        r_bond_types = {} # Map the R group to bond type
        
        # Go through each atom and: i) get index of attachment atoms, ii) tag all non-R atoms
        for atom in atoms:
            # if R atom
            if "*" in atom.GetSmarts():
                # Get the index of the atom it's attached to
                neighbors = atom.GetNeighbors()
                assert len(neighbors) == 1
                neighbor_idx = neighbors[0].GetIdx()
                r_tag = atom.GetSmarts().strip("[]").replace(":", "") # *1, *2, ...
                neighbor_map[r_tag] = neighbor_idx
                # Tag it as non-core atom
                atom.SetBoolProp('core', False)
                # Create a map R --> bond type
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_idx)
                r_bond_types[r_tag] = bond.GetBondType()
            # If not R atom
            else:
                # Tag it as a core atom
                atom.SetBoolProp('core', True)
        
        # Use the map created to tag attachment points
        for atom in atoms:
            if atom.GetIdx() in neighbor_map.values():
                r_tags = [k for k, v in neighbor_map.items() if v == atom.GetIdx()]
                atom.SetProp('R', ''.join(r_tags))
            else:
                atom.SetProp('R', '')
        
        return mol, r_bond_types


    def remove_wildcard_atoms(self, rwmol):
        """
        removes wildcard atoms from an RDKit Mol
        """
        indicies = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
        while len(indicies) > 0:
            rwmol.RemoveAtom(indicies[0])
            indicies = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
        Chem.SanitizeMol(rwmol, Chem.SanitizeFlags.SANITIZE_ALL)
        
        return rwmol
    
    
    def parse_polymer_rules(self, rules):
        """
        parses the polymer rules provided in the polymer input string
        """
        polymer_info = []
        counter = Counter() # Used for validating the input
        
        # Check if deg of polymerisation is provided
        if '~' in rules[-1]:
            Xn = float(rules[-1].split('~')[1])
            rules[-1] = rules[-1].split('~')[0]
        else:
            Xn = 1
        
        for rule in rules:
            # Handle the edge case where we have no rules, and rule is an empty string
            if rule == "":
                continue
            # QC of the input string
            if len(rule.split(":")) != 3:
                raise ValueError(f"Incorrect format for input information {rule}")
            idx1, idx2 = rule.split(':')[0].split('-')
            w12 = float(rule.split(':')[1]) # Weight for bond R_idx1 -> R_idx2
            w21 = float(rule.split(':')[2]) # Weight for bond R_idx2 -> R_idx1
            polymer_info.append((idx1, idx2, w12, w21))
            counter[idx1] += float(w21)
            counter[idx2] += float(w12)
            
        # Validate input: sum of the incoming weights should be 1 for each vertex
        for k, v in counter.items():
            if np.isclose(v, 1.0) is False:
                raise ValueError(f"Sum of weights of incoming stochastic edges should be 1 -- found {v} for [*:{k}]")
        
        return polymer_info, 1. + np.log10(Xn)
    
    def __post_init__(self, extra_atom_fdim: int = 0, extra_bond_fdim: int = 0, degree_of_poly: int = 1, polymer_info: list = []):
        super().__post_init__()

        self.extra_atom_fdim = extra_atom_fdim
        self.extra_bond_fdim = extra_bond_fdim
        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim
        self.degree_of_poly = degree_of_poly
        self.polymer_info = polymer_info
        
    def __call__(
        self,
        polymer: Polymer,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        mol = polymer.mol
        rules = polymer.edges
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()
        self.polymer_info, self.degree_of_poly = self.parse_polymer_rules(rules)
        # Make the molecule editable
        rwmol = Chem.rdchem.RWMol(mol)
        # tag (i) attachment atoms and (ii) atoms for which features need to be computed
        # also get the map of R groups to bond types, e.g. r_bond_types[*1] -> SINGLE
        rwmol, r_bond_types = self.tag_atoms_in_repeating_unit(rwmol) 
        # -----------------
        # Get atom features
        # -----------------
        # for all 'core' atoms, i.e. not R groups, as tagged before. This is done here to ensure that atoms linked to R groups
        # have the correct saturation
        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
            V_w = np.zeros((1, self.atom_fdim), dtype=np.single).flatten()
        else:
            V = np.array([self.atom_featurizer(a) for a in rwmol.GetAtoms() if a.GetBoolProp('core') is True], dtype=np.single)
            V_w = np.array([a.GetDoubleProp('w_frag') for a in rwmol.GetAtoms() if a.GetBoolProp('core') is True], dtype=np.single).flatten()
        # Concatenate the extra atoms features to the atom features (V)
        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))
        # Check to ensure that the number of atoms equals the length of the extra atom features
        n_atoms = len(V)
        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )        
        # Remove R groups -> now atoms in the RDKit Mol object have the same order as V
        rwmol = self.remove_wildcard_atoms(rwmol)     
        # Initialize atom to bond mapping for each atom
        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]
        # Initalize bond weight array
        E_w = []
        # ---------------------------------------
        # Get bond features for separate monomers
        # ---------------------------------------
        # Create counter for the number of bonds (i)
        i = 0
        for bond in rwmol.GetBonds():
            if bond is None:
                continue
            x_e = self.bond_featurizer(bond)
            if bond_features_extra is not None:
                x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single)
            # Update the index mappings
            E[i : i + 2] = x_e
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])
            E_w.extend([1.0, 1.0])
            # Increase the total number of bonds by 2
            i += 2
        # -------------------------------------------------------
        # Get the bond features for bonds between repeating units
        # -------------------------------------------------------
        # we duplicate the monomers present to allow (i) creating bonds that exist already within the same molecule and 
        # (ii) collect the correct bond features, e.g., for bonds that would otherwise be considered in a ring when they are not,
        # when e.g. creating a bond between two atoms in the same ring.
        rwmol_copy = deepcopy(rwmol)
        _ = [a.SetBoolProp('OriginalMol', True) for a in rwmol.GetAtoms()]
        _ = [a.SetBoolProp('OriginalMol', False) for a in rwmol_copy.GetAtoms()]
        # Create an editable combined Mol
        cm = Chem.CombineMols(rwmol, rwmol_copy)
        cm = Chem.RWMol(cm)
        # The combined molecule now has double the bonds, therefore we must duplicate the extra atom bond features not relating to the
        # bonds between monomers then append the extra bond features relating to the bonds between monomers
        if bond_features_extra is not None:
            n_bonds = rwmol.GetNumBonds()
            combined_bond_features_extra = np.concatenate((bond_features_extra[:n_bonds],
                                                            bond_features_extra[:n_bonds],
                                                            bond_features_extra[n_bonds:],
                                                            bond_features_extra[n_bonds:]))
        # For all possible bonds between monomers:
        # add bond -> compute bond features -> add bond to list -> remove bond
        for r1, r2, w_bond12, w_bond21 in self.polymer_info:
            # Get the index of the attachment atoms
            a1 = None # The index of atom 1 in rwmol
            a2 = None # The index of atom 1 in rwmol -> to be used by MolGraph
            _a2 = None # The index of atom 1 in cm -> to be used by RDKit
            for atom in cm.GetAtoms():
                # Take a1 from the fragement in the original molecule object
                if f"*{r1}" in atom.GetProp('R') and atom.GetBoolProp('OriginalMol') is True:
                    a1 = atom.GetIdx()
                # take _a2 from a fragment in the copied molecule object, but a2 from the original
                if f"*{r2}" in atom.GetProp('R'):
                    if atom.GetBoolProp('OriginalMol') is True:
                        a2 = atom.GetIdx()
                    elif atom.GetBoolProp('OriginalMol') is False:
                        _a2 = atom.GetIdx()
            # Check to ensure bond definitions are valid
            if a1 is None:
                raise ValueError(f"Cannot find atom attached to [*:{r1}]")
            if a2 is None or _a2 is None:
                raise ValueError(f"Cannot find atom attached to [*:{r2}]")
            
            # Create a bond
            bond_order1 = r_bond_types[f'*{r1}']
            bond_order2 = r_bond_types[f'*{r2}']
            if bond_order1 != bond_order2:
                raise ValueError(f"Two atoms are trying to be bonded with different bond types: {bond_order1} vs {bond_order2}")
            cm.AddBond(a1, _a2, order=bond_order1)
            Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL)
            
            # Get the bond object and its features
            bond = cm.GetBondBetweenAtoms(a1, _a2)
            x_e = self.bond_featurizer(bond)
            if bond_features_extra is not None:
                x_e = np.concatenate((x_e, combined_bond_features_extra[bond.GetIdx()]), dtype=np.single)
            # Update the index mappings
            E[i : i + 2] = x_e
            edge_index[0].extend([a1, a2])
            edge_index[1].extend([a2, a1])
            E_w.extend([w_bond12, w_bond21])
            # Increase the total number of bonds by 2
            i += 2
            
            # Remove the bond
            cm.RemoveBond(a1, _a2)
            Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL)
            
        # Check to ensure that the number of bonds equals the length of the number of extra bond features
        if bond_features_extra is not None and len(bond_features_extra) != i / 2:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {i} and {len(bond_features_extra)}, respectively"
            )  
        # Reverse the edge indexes
        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        # Convert the edge index to a numpy array of type int
        edge_index = np.array(edge_index, int)
        E_w = np.array(E_w, float)

        return MolGraph(V, E, V_w, E_w, edge_index, rev_edge_index, self.degree_of_poly)
            
        