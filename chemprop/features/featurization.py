from typing import List, Tuple, Union
from itertools import zip_longest
import logging

from rdkit import Chem
import torch
import numpy as np
from collections import deque

from chemprop.rdkit import make_mol

class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.
    """
    def __init__(self) -> None:

        # Atom feature sizes
        self.MAX_ATOMIC_NUM = 100
        self.ATOM_FEATURES = {
            'atomic_num': list(range(self.MAX_ATOMIC_NUM)),
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
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM = sum(len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM = 14
        self.EXTRA_BOND_FDIM = 0
        self.REACTION_MODE = None
        self.EXPLICIT_H = False
        self.REACTION = False
        self.ADDING_H = False
        self.KEEP_ATOM_MAP = False

# Create a global parameter object for reference throughout this module
PARAMS = Featurization_parameters()


def reset_featurization_parameters(logger: logging.Logger = None) -> None:
    """
    Function resets feature parameter values to defaults by replacing the parameters instance.
    """
    if logger is not None:
        debug = logger.debug
    else:
        debug = print
    debug('Setting molecule featurization parameters to default.')
    global PARAMS
    PARAMS = Featurization_parameters()


def get_atom_fdim(overwrite_default_atom: bool = False, is_reaction: bool = False) -> int:
    """
    Gets the dimensionality of the atom feature vector.

    :param overwrite_default_atom: Whether to overwrite the default atom descriptors.
    :param is_reaction: Whether to add :code:`EXTRA_ATOM_FDIM` for reaction input when :code:`REACTION_MODE` is not None.
    :return: The dimensionality of the atom feature vector.
    """
    if PARAMS.REACTION_MODE:
        return (not overwrite_default_atom) * PARAMS.ATOM_FDIM + is_reaction * PARAMS.EXTRA_ATOM_FDIM
    else:
        return (not overwrite_default_atom) * PARAMS.ATOM_FDIM + PARAMS.EXTRA_ATOM_FDIM


def set_explicit_h(explicit_h: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with explicit Hs.

    :param explicit_h: Boolean whether to keep explicit Hs from input.
    """
    PARAMS.EXPLICIT_H = explicit_h

def set_adding_hs(adding_hs: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with adding the Hs to them.

    :param adding_hs: Boolean whether to add Hs to the molecule.
    """
    PARAMS.ADDING_H = adding_hs

def set_keeping_atom_map(keeping_atom_map: bool) -> None:
    """
    Sets whether RDKit molecules keep the original atom mapping.

    :param keeping_atom_map: Boolean whether to keep the original atom mapping.
    """
    PARAMS.KEEP_ATOM_MAP = keeping_atom_map

def set_reaction(reaction: bool, mode: str) -> None:
    """
    Sets whether to use a reaction or molecule as input and adapts feature dimensions.
 
    :param reaction: Boolean whether to except reactions as input.
    :param mode: Reaction mode to construct atom and bond feature vectors.

    """
    PARAMS.REACTION = reaction
    if reaction:
        PARAMS.EXTRA_ATOM_FDIM = PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1
        PARAMS.EXTRA_BOND_FDIM = PARAMS.BOND_FDIM
        PARAMS.REACTION_MODE = mode
        
def is_explicit_h(is_mol: bool = True) -> bool:
    r"""Returns whether to retain explicit Hs (for reactions only)"""
    if not is_mol:
        return PARAMS.EXPLICIT_H
    return False


def is_adding_hs(is_mol: bool = True) -> bool:
    r"""Returns whether to add explicit Hs to the mol (not for reactions)"""
    if is_mol:
        return PARAMS.ADDING_H
    return False


def is_keeping_atom_map(is_mol: bool = True) -> bool:
    r"""Returns whether to keep the original atom mapping (not for reactions)"""
    if is_mol:
        return PARAMS.KEEP_ATOM_MAP
    return True


def is_reaction(is_mol: bool = True) -> bool:
    r"""Returns whether to use reactions as input"""
    if is_mol:
        return False
    if PARAMS.REACTION: #(and not is_mol, checked above)
        return True
    return False


def reaction_mode() -> str:
    r"""Returns the reaction mode"""
    return PARAMS.REACTION_MODE


def set_extra_atom_fdim(extra):
    """Change the dimensionality of the atom feature vector."""
    PARAMS.EXTRA_ATOM_FDIM = extra


def get_bond_fdim(atom_messages: bool = False,
                  overwrite_default_bond: bool = False,
                  overwrite_default_atom: bool = False,
                  is_reaction: bool = False) -> int:
    """
    Gets the dimensionality of the bond feature vector.

    :param atom_messages: Whether atom messages are being used. If atom messages are used,
                          then the bond feature vector only contains bond features.
                          Otherwise it contains both atom and bond features.
    :param overwrite_default_bond: Whether to overwrite the default bond descriptors.
    :param overwrite_default_atom: Whether to overwrite the default atom descriptors.
    :param is_reaction: Whether to add :code:`EXTRA_BOND_FDIM` for reaction input when :code:`REACTION_MODE:` is not None
    :return: The dimensionality of the bond feature vector.
    """

    if PARAMS.REACTION_MODE:
        return (not overwrite_default_bond) * PARAMS.BOND_FDIM + is_reaction * PARAMS.EXTRA_BOND_FDIM + \
            (not atom_messages) * get_atom_fdim(overwrite_default_atom=overwrite_default_atom, is_reaction=is_reaction)
    else:
        return (not overwrite_default_bond) * PARAMS.BOND_FDIM + PARAMS.EXTRA_BOND_FDIM + \
            (not atom_messages) * get_atom_fdim(overwrite_default_atom=overwrite_default_atom, is_reaction=is_reaction)


def set_extra_bond_fdim(extra):
    """Change the dimensionality of the bond feature vector."""
    PARAMS.EXTRA_BOND_FDIM = extra


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]  # scaled to about the same range as other features
        if functional_groups is not None:
            features += functional_groups
    return features


def atom_features_zeros(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom containing only the atom number information.

    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            [0] * (PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1) #set other features to zero
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1)
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
    return fbond


def map_reac_to_prod(mol_reac: Chem.Mol, mol_prod: Chem.Mol):
    """
    Build a dictionary of mapping atom indices in the reactants to the products.

    :param mol_reac: An RDKit molecule of the reactants.
    :param mol_prod: An RDKit molecule of the products.
    :return: A dictionary of corresponding reactant and product atom indices.
    """
    only_prod_ids = []
    prod_map_to_id = {}
    mapnos_reac = set([atom.GetAtomMapNum() for atom in mol_reac.GetAtoms()]) 
    for atom in mol_prod.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            prod_map_to_id[mapno] = atom.GetIdx()
            if mapno not in mapnos_reac:
                only_prod_ids.append(atom.GetIdx()) 
        else:
            only_prod_ids.append(atom.GetIdx())
    only_reac_ids = []
    reac_id_to_prod_id = {}
    for atom in mol_reac.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            try:
                reac_id_to_prod_id[atom.GetIdx()] = prod_map_to_id[mapno]
            except KeyError:
                only_reac_ids.append(atom.GetIdx())
        else:
            only_reac_ids.append(atom.GetIdx())
    return reac_id_to_prod_id, only_prod_ids, only_reac_ids


'''

Pretraining involved functions

'''

def mask_atom_features(atom_features: List[Union[bool, int, float]],overwrite_default_atom: bool = False) -> List[Union[bool, int, float]]:
    '''

    :param atom_features: The original feature of atoms
    :return: a list of 0, which replaces all the original feature of atoms
    '''

    atom_features_len = len(atom_features)
    assert atom_features_len == get_atom_fdim(overwrite_default_atom)
    masked_atom_features = [0] * atom_features_len

    return masked_atom_features

def mask_bond_atom_features(bond_features: List[Union[bool, int, float]], atom_messages: bool = False, overwrite_default_atom: bool = False,overwrite_default_bond: bool = False) -> List[Union[bool, int, float]]:
    '''

    :param bond_features: The original feature of bonds
    :return: a list of 0, which replaces all the original feature of atoms
    '''

    atom_features_len = get_atom_fdim(overwrite_default_atom)
    bond_features_len = len(bond_features)
    assert bond_features_len == get_bond_fdim(atom_messages=atom_messages,overwrite_default_atom=overwrite_default_atom,overwrite_default_bond=overwrite_default_bond)
    mask_bond_atom_features = [0] * atom_features_len
    mask_bond_atom_features.extend(bond_features[atom_features_len:])

    return mask_bond_atom_features

def mask_bond_features(bond_features: List[Union[bool, int, float]], atom_messages: bool = False, overwrite_default_atom: bool = False,overwrite_default_bond: bool = False) -> List[Union[bool, int, float]]:
    '''

    :param bond_features: The original feature of bonds
    :return: a list of 0, which replaces all the original feature of atoms
    '''

    atom_features_len = get_atom_fdim(overwrite_default_atom)
    bond_features_len = len(bond_features)
    assert bond_features_len == get_bond_fdim(atom_messages=atom_messages,overwrite_default_atom=overwrite_default_atom,overwrite_default_bond=overwrite_default_bond)
    mask_bond_features = [0] * bond_features_len


    return mask_bond_features


class MolGraph:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:

    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
    * :code:`overwrite_default_atom_features`: A boolean to overwrite default atom descriptors.
    * :code:`overwrite_default_bond_features`: A boolean to overwrite default bond descriptors.
    * :code:`is_mol`: A boolean whether the input is a molecule.
    * :code:`is_reaction`: A boolean whether the molecule is a reaction.
    * :code:`is_explicit_h`: A boolean whether to retain explicit Hs (for reaction mode).
    * :code:`is_adding_hs`: A boolean whether to add explicit Hs (not for reaction mode).
    * :code:`reaction_mode`:  Reaction mode to construct atom and bond feature vectors.
    * :code:`b2br`: A mapping from f_bonds to real bonds in molecule recorded in targets.
    """

    def __init__(self, mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
                 atom_features_extra: np.ndarray = None,
                 bond_features_extra: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False):
        """
        :param mol: A SMILES or an RDKit molecule.
        :param atom_features_extra: A list of 2D numpy array containing additional atom features to featurize the molecule.
        :param bond_features_extra: A list of 2D numpy array containing additional bond features to featurize the molecule.
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features instead of concatenating.
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features instead of concatenating.
        """
        self.is_mol = is_mol(mol)
        self.is_reaction = is_reaction(self.is_mol)
        self.is_explicit_h = is_explicit_h(self.is_mol)
        self.is_adding_hs = is_adding_hs(self.is_mol)
        self.is_keeping_atom_map = is_keeping_atom_map(self.is_mol)
        self.reaction_mode = reaction_mode()
        
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            if self.is_reaction:
                mol = (make_mol(mol.split(">")[0], self.is_explicit_h, self.is_adding_hs, self.is_keeping_atom_map), make_mol(mol.split(">")[-1], self.is_explicit_h, self.is_adding_hs, self.is_keeping_atom_map)) 
            else:
                mol = make_mol(mol, self.is_explicit_h, self.is_adding_hs, self.is_keeping_atom_map)

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features

        if not self.is_reaction:
            # Get atom features
            self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
            if atom_features_extra is not None:
                if overwrite_default_atom_features:
                    self.f_atoms = [descs.tolist() for descs in atom_features_extra]
                else:
                    self.f_atoms = [f_atoms + descs.tolist() for f_atoms, descs in zip(self.f_atoms, atom_features_extra)]

            self.n_atoms = len(self.f_atoms)
            if atom_features_extra is not None and len(atom_features_extra) != self.n_atoms:
                raise ValueError(f'The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra atom features')

            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Initialize f_bonds to real bonds mapping for each bond
            self.b2br = np.zeros([len(mol.GetBonds()), 2])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue

                    f_bond = bond_features(bond)
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr

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
                    self.b2br[bond.GetIdx(), :] = [self.n_bonds, self.n_bonds + 1]
                    self.n_bonds += 2

            if bond_features_extra is not None and len(bond_features_extra) != self.n_bonds / 2:
                raise ValueError(f'The number of bonds in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra bond features')

        else: # Reaction mode
            if atom_features_extra is not None:
                raise NotImplementedError('Extra atom features are currently not supported for reactions')
            if bond_features_extra is not None:
                raise NotImplementedError('Extra bond features are currently not supported for reactions')

            mol_reac = mol[0]
            mol_prod = mol[1]
            ri2pi, pio, rio = map_reac_to_prod(mol_reac, mol_prod)
           
            # Get atom features
            if self.reaction_mode in ['reac_diff','prod_diff', 'reac_prod']:
                #Reactant: regular atom features for each atom in the reactants, as well as zero features for atoms that are only in the products (indices in pio)
                f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [atom_features_zeros(mol_prod.GetAtomWithIdx(index)) for index in pio]
                
                #Product: regular atom features for each atom that is in both reactants and products (not in rio), other atom features zero,
                #regular features for atoms that are only in the products (indices in pio)
                f_atoms_prod = [atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])) if atom.GetIdx() not in rio else
                                atom_features_zeros(atom) for atom in mol_reac.GetAtoms()] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
            else: #balance
                #Reactant: regular atom features for each atom in the reactants, copy features from product side for atoms that are only in the products (indices in pio)
                f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
                
                #Product: regular atom features for each atom that is in both reactants and products (not in rio), copy features from reactant side for
                #other atoms, regular features for atoms that are only in the products (indices in pio)
                f_atoms_prod = [atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])) if atom.GetIdx() not in rio else
                                atom_features(atom) for atom in mol_reac.GetAtoms()] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]

            if self.reaction_mode in ['reac_diff', 'prod_diff', 'reac_diff_balance', 'prod_diff_balance']:
                f_atoms_diff = [list(map(lambda x, y: x - y, ii, jj)) for ii, jj in zip(f_atoms_prod, f_atoms_reac)]
            if self.reaction_mode in ['reac_prod', 'reac_prod_balance']:
                self.f_atoms = [x+y[PARAMS.MAX_ATOMIC_NUM+1:] for x,y in zip(f_atoms_reac, f_atoms_prod)]
            elif self.reaction_mode in ['reac_diff', 'reac_diff_balance']:
                self.f_atoms = [x+y[PARAMS.MAX_ATOMIC_NUM+1:] for x,y in zip(f_atoms_reac, f_atoms_diff)]
            elif self.reaction_mode in ['prod_diff', 'prod_diff_balance']:
                self.f_atoms = [x+y[PARAMS.MAX_ATOMIC_NUM+1:] for x,y in zip(f_atoms_prod, f_atoms_diff)]
            self.n_atoms = len(self.f_atoms)
            n_atoms_reac = mol_reac.GetNumAtoms()

            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    if a1 >= n_atoms_reac and a2 >= n_atoms_reac: # Both atoms only in product
                        bond_prod = mol_prod.GetBondBetweenAtoms(pio[a1 - n_atoms_reac], pio[a2 - n_atoms_reac])
                        if self.reaction_mode in ['reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance']:
                            bond_reac = bond_prod
                        else:
                            bond_reac = None
                    elif a1 < n_atoms_reac and a2 >= n_atoms_reac: # One atom only in product
                        bond_reac = None
                        if a1 in ri2pi.keys():
                            bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[a1], pio[a2 - n_atoms_reac])
                        else:
                            bond_prod = None # Atom atom only in reactant, the other only in product
                    else:
                        bond_reac = mol_reac.GetBondBetweenAtoms(a1, a2)
                        if a1 in ri2pi.keys() and a2 in ri2pi.keys():
                            bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[a1], ri2pi[a2]) #Both atoms in both reactant and product
                        else:
                            if self.reaction_mode in ['reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance']:
                                if a1 in ri2pi.keys() or a2 in ri2pi.keys():
                                    bond_prod = None # One atom only in reactant
                                else:
                                    bond_prod = bond_reac # Both atoms only in reactant
                            else:    
                                bond_prod = None # One or both atoms only in reactant

                    if bond_reac is None and bond_prod is None:
                        continue

                    f_bond_reac = bond_features(bond_reac)
                    f_bond_prod = bond_features(bond_prod)
                    if self.reaction_mode in ['reac_diff', 'prod_diff', 'reac_diff_balance', 'prod_diff_balance']:
                        f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
                    if self.reaction_mode in ['reac_prod', 'reac_prod_balance']:
                        f_bond = f_bond_reac + f_bond_prod
                    elif self.reaction_mode in ['reac_diff', 'reac_diff_balance']:
                        f_bond = f_bond_reac + f_bond_diff
                    elif self.reaction_mode in ['prod_diff', 'prod_diff_balance']:
                        f_bond = f_bond_prod + f_bond_diff
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
    * :code:`b2br`: (Optional): A mapping from f_bonds to real bonds in molecule recorded in targets.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.mol_graphs = mol_graphs
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

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.tensor(f_atoms, dtype=torch.float)
        self.f_bonds = torch.tensor(f_bonds, dtype=torch.float)
        self.a2b = torch.tensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)], dtype=torch.long)
        self.b2a = torch.tensor(b2a, dtype=torch.long)
        self.b2revb = torch.tensor(b2revb, dtype=torch.long)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages
        self.b2br = None  # only needed in predictions of atomic/bond targets

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
                                                                   torch.Tensor, torch.Tensor, torch.Tensor,
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
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages=atom_messages,
                                                     overwrite_default_atom=self.overwrite_default_atom_features,
                                                     overwrite_default_bond=self.overwrite_default_bond_features):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.Tensor:
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

    def get_a2a(self) -> torch.Tensor:
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

    def get_b2br(self) -> torch.Tensor:
        """
        Computes (if necessary) and returns a mapping from f_bonds to real bonds in molecule recorded in targets.

        :return: A PyTorch tensor containing the mapping from f_bonds to real bonds in molecule recorded in targets.
        """
        if self.b2br is None:
            n_bonds = 1 # number of bonds (start at 1 b/c need index 0 as padding)
            b2br = []
            for mol_graph in self.mol_graphs:
                b2br.append(mol_graph.b2br + n_bonds)
                n_bonds += mol_graph.n_bonds
            b2br = np.concatenate(b2br, axis=0)
            self.b2br = torch.tensor(b2br, dtype=torch.long)

        return self.b2br

def mol2graph(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
              atom_features_batch: List[np.array] = (None,),
              bond_features_batch: List[np.array] = (None,),
              overwrite_default_atom_features: bool = False,
              overwrite_default_bond_features: bool = False
              ) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param atom_features_batch: A list of 2D numpy array containing additional atom features to featurize the molecule.
    :param bond_features_batch: A list of 2D numpy array containing additional bond features to featurize the molecule.
    :param overwrite_default_atom_features: Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating.
    :param overwrite_default_bond_features: Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating.
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraph([MolGraph(mol, af, bf,
                                   overwrite_default_atom_features=overwrite_default_atom_features,
                                   overwrite_default_bond_features=overwrite_default_bond_features)
                          for mol, af, bf
                          in zip_longest(mols, atom_features_batch, bond_features_batch)])

def is_mol(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]) -> bool:
    """Checks whether an input is a molecule or a reaction

    :param mol: str, RDKIT molecule or tuple of molecules.
    :return: Whether the supplied input corresponds to a single molecule.
    """

    if isinstance(mol, str) and ">" not in mol:
        return True
    elif isinstance(mol, Chem.Mol):
        return True
    return False

'''

Pretraining involved featurization

'''


class MolGraphPretrain(object):
    """
    A :class:`MolGraph_pretrain` represents the graph structure and featurization of a single molecule used for pretraining.
    For pretraining, the reaction mode is not considered yet
    There are three main pretraining related functions:
    1. masked_atom_pretraining(self) It will mask certain portion of atoms by masking their atom features and correlating bond features
    It can be used for atom attibute pretraining (local) as well as contrastive learning pretraining(global).
    2. bond_deletion_complete(self) It will delete certain bonds from the molecule, the connection for message passing is broken after this operation
    It will be used for contrastive learning pretraining(global)
    3. subgraph_deletion(self,center) It will delete both atoms and bonds which belong to a subgraph of the molecule
    It will be used for contrastive learning pretraining(global)

    All three functions will change the attributes of the MolGraphPretrain object. It is irreversible and can only be applied once.
    Before the operation, can firstly use deepcopy to get multiple copies of the raw MolGraphPretrain object

    A MolGraph computes the following attributes:


    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
    * :code:`overwrite_default_atom_features`: A boolean to overwrite default atom descriptors.
    * :code:`overwrite_default_bond_features`: A boolean to overwrite default bond descriptors.
    * :code:`is_mol`: A boolean whether the input is a molecule.
    * :code:`is_explicit_h`: A boolean whether to retain explicit Hs (for reaction mode).
    * :code:`is_adding_hs`: A boolean whether to add explicit Hs (not for reaction mode).
    * :code:`b2br`: A mapping from f_bonds to real bonds in molecule recorded in targets.
    """

    def __init__(self, mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
                 atom_features_extra: np.ndarray = None,
                 bond_features_extra: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False,
                 mask_atom_pre_auto: bool = True,
                 mask_atom_pre_percent: float = 0.0,
                 mask_bond_pre_auto: bool = True,
                 mask_bond_pre_percent: float = 0.0,
                 mask_subgraph_pre_percent: float = 0.0):
        """
        For pretraining, the reaction mode is not considered yet
        :param mol: A SMILES or an RDKit molecule.
        :param atom_features_extra: A list of 2D numpy array containing additional atom features to featurize the molecule.
        :param bond_features_extra: A list of 2D numpy array containing additional bond features to featurize the molecule.
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features instead of concatenating.
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features instead of concatenating.
        :param mask_atom_pre_auto: Boolean to state whether the mask atom pretrain is done by auto masking with random.
        :param mask_atom_pre_percent: Float to state the percentage of atoms to mask in mask atom pretrain.
        :param mask_bond_pre_auto: Boolean to state whether the bond deletion  is done by auto delete with random.
        :param mask_bond_pre_percent: Float to state the percentage of bonds to mask in bond deletion pretrain.
        :param mask_subgraph_pre_percent: Float to state the percentage of subgraph to mask in subgraph deletion pretrain.
        """
        self.is_mol = is_mol(mol)
        self.is_reaction = is_reaction(self.is_mol)
        self.is_explicit_h = is_explicit_h(self.is_mol)
        self.is_adding_hs = is_adding_hs(self.is_mol)

        # arguments for MA_pretraining
        self.mask_atom_auto = mask_atom_pre_auto
        self.mask_atom_percentage = mask_atom_pre_percent
        self.masked_atom_label_list = []
        self.final_masked_atom_index = None
        # arguments for bond_deletion pretraining
        self.mask_bond_auto = mask_bond_pre_auto
        self.mask_bond_percentage = mask_bond_pre_percent
        # arguments for sub_graph_deletion pretraining
        self.mask_subgraph_pre_percent = mask_subgraph_pre_percent

        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            mol = make_mol(mol, self.is_explicit_h, self.is_adding_hs, self.is_keeping_atom_map_list)
        self.smiles = Chem.MolToSmiles(mol)

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features
        self.a2a = []  # only compute when necessary
        self.atom_num_label = []  # the atom num label used for masking atom pretraining
        self.masked_atom_label_list = []  # the final masked atom num label used for masking atom pretraining

        # Get atom features
        self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]

        self.atom_num_label = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        if atom_features_extra is not None:
            if overwrite_default_atom_features:
                self.f_atoms = [descs.tolist() for descs in atom_features_extra]
            else:
                self.f_atoms = [f_atoms + descs.tolist() for f_atoms, descs in
                                zip(self.f_atoms, atom_features_extra)]

        self.n_atoms = len(self.f_atoms)
        if atom_features_extra is not None and len(atom_features_extra) != self.n_atoms:
            raise ValueError(f'The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of '
                             f'the extra atom features')

        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Initialize f_bonds to real bonds mapping for each bond
        self.b2br = np.zeros([len(mol.GetBonds()), 2])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)
                if bond_features_extra is not None:
                    descr = bond_features_extra[bond.GetIdx()].tolist()
                    if overwrite_default_bond_features:
                        f_bond = descr
                    else:
                        f_bond += descr

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
                self.b2br[bond.GetIdx(), :] = [self.n_bonds, self.n_bonds + 1]
                self.n_bonds += 2

        if bond_features_extra is not None and len(bond_features_extra) != self.n_bonds / 2:
            raise ValueError(f'The number of bonds in {Chem.MolToSmiles(mol)} is different from the length of '
                             f'the extra bond features')

    def masked_atom_pretraining(self):
        '''
        The atom feature is masked by replacing original feature with 0
        The bond feature's atom part will be replaced with 0 if the atom the bond originates from is masked
        :return: No return. Will just modify MolGraphPretrain object into masked atom version.
        :Note: The MolGraphPretrain(object) will be modified, and for different pretraining operation should init a new object then use the operation function
        '''

        if self.mask_atom_auto:

            self.masked_atom_index = torch.randperm(len(self.f_atoms))[
                                     :int(len(self.f_atoms) * self.mask_atom_percentage)]
            self.masked_atom_index = self.masked_atom_index.tolist()
            self.masked_atom_index = sorted(self.masked_atom_index)

            self.f_atoms = [
                self.f_atoms[idx] if idx not in self.masked_atom_index else mask_atom_features(self.f_atoms[idx]) for
                idx in range(len(self.f_atoms))]

            # find bond index where the atom the bond originates from is masked
            bond_index_to_be_masked = [i for i, val in enumerate(self.b2a) if val in self.masked_atom_index]

            # mask the atom feature in the bond feature
            self.f_bonds = [
                self.f_bonds[idx] if idx not in bond_index_to_be_masked else mask_bond_atom_features(self.f_bonds[idx])
                for
                idx in range(len(self.f_bonds))]

            self.final_masked_atom_index = self.masked_atom_index

            self.masked_atom_label_list = [
                self.atom_num_label[idx] for idx in self.final_masked_atom_index]




    def bond_deletion_complete(self):
        '''
        This operation is to delete certain percentage of the bond completely (i.e., we have directional bond, this will
        delete both the direction).

        :return:
        '''
        if self.mask_bond_auto:
            # Get random bond deletion index from the nondirectional bond matrix b2br
            number_of_undirectional_bonds = len(self.b2br)
            self.masked_complete_bond_index = torch.randperm(number_of_undirectional_bonds)[
                                              :int(number_of_undirectional_bonds * self.mask_bond_percentage)]
            self.masked_complete_bond_index = self.masked_complete_bond_index.tolist()
            self.masked_complete_bond_index = sorted(self.masked_complete_bond_index)

            # need to operate on f_bonds
            masked_directional_bond_index = self.b2br[self.masked_complete_bond_index]
            masked_directional_bond_index_flattened = [item for sublist in masked_directional_bond_index for item in
                                                       sublist]

            # mask the bond feature, this operation is important, since if we directly delete the bond features then all the bond index will need to be updated
            # recursively. Although it may not be that memory efficient but it can be more computational efficient

            self.f_bonds = [
                self.f_bonds[idx] if idx not in masked_directional_bond_index_flattened
                else mask_bond_features(self.f_bonds[idx], overwrite_default_atom=self.overwrite_default_atom_features,
                                        overwrite_default_bond=self.overwrite_default_bond_features) for idx in
                range(len(self.f_bonds))]
            # when considering about how to update the connectivity of `a2b`, `b2a`, `b2revb` and, `b2br`, we need to look into the message passing process, and the batch graph process
            '''
            when considering message passing from bonds, we have the following:
            # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
            # message      a_message = sum(nei_a_message)      rev_message
            nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden

            For nei_a_message in first time update it is fine not to change a2b as the f_bonds with deleleted bonds are replaced with 0s
            However, these f_bond will have information if no operation is done to b2a
            Then all the deleleted bonds will have message which we don't want. 

            So in order to make the deleted bond perform correctly as a deleted bond we need to change the connectivity.
            Make the all the deleted bond either refer to f_atoms[0], f_bonds[0] message[0], and a_message[0]. So it is 
            always empty
            a2b (a list of list) we need to delete the bonds, then such a bond will be relaced to 0 in batch graph construction. 
            b2a, (a list) for the index of the bond which is deleted, it should be refer to 0 which is the padding a_message. it will be done in batch graph construction as well.
            b2revb, (a list) if the revbond is deleted, it should be refer to 0 which is the padding of message. it will be done in batch graph construction as well. Although it may be redundant, it is safer.
            b2br, (a list) it is only used if self.is_atom_bond_targets == Ture. Since all the message at the deleted bond is 0s and f_bonds of deleted bonds are 0s. This one no need to change. It will be 0s.            
            '''
            # it will delete the bonds, and the bonds index will be replaced with 0 in batch graph construction
            self.a2b = [[bond for bond in bonds if bond not in masked_directional_bond_index_flattened] for bonds in
                        self.a2b]

            # for the bond which is deleted it will originates from an empty atom, a special token del is places first, then it will be replaced with 0 in batch graph construction
            self.b2a = ['del' if index in masked_directional_bond_index_flattened else value for index, value in
                        enumerate(self.b2a)]

            # for the bond which is deleted it will originates from an empty atom, a special token del is places first, then it will be replaced with 0 in batch graph construction
            self.b2revb = ['del' if value in masked_directional_bond_index_flattened else value for value in
                           self.b2revb]

            self.final_deleted_bond_index = masked_directional_bond_index_flattened


    def subgraph_deletion(self, center):
        '''
        Remove a subgraph meaning mask the atom to be removed and remove all the bonds connected to it.

        :param center: The start atom to remove

        :return: it is just an operation on this object
        '''

        num_of_atoms_of_removed_subgraph = np.floor(len(self.f_atoms) * self.mask_subgraph_pre_percent)
        start_center_atom_idx = center
        atom_idx_to_remove = [start_center_atom_idx]
        queue = deque([start_center_atom_idx])  # Queue for BFS

        # Iterate over each atom's incoming bonds
        for incoming_bonds in self.a2b:
            # Map each bond to its corresponding atom
            neighboring_atoms = [self.b2a[b] for b in incoming_bonds]
            # Append the list of neighboring atoms to a2a
            self.a2a.append(neighboring_atoms)  # For each atom_idx get the neighboring atom index

        # BFS to get all the atom_idx_to_remove.
        while len(atom_idx_to_remove) < num_of_atoms_of_removed_subgraph:
            # Get the next atom from the queue
            current_atom = queue.popleft()

            # Add all unvisited neighboring atoms to the queue and to atom_idx_to_remove
            for neighbor in self.a2a[current_atom]:
                if neighbor not in atom_idx_to_remove:
                    atom_idx_to_remove.append(neighbor)
                    queue.append(neighbor)

            # If there are no more atoms to visit and we still haven't reached
            # num_of_atoms_of_removed_subgraph, we break the loop to avoid an infinite loop.
            if not queue:
                break

        # Find all the bonds to delete
        bonds_to_delete = []
        for atom in atom_idx_to_remove:

            for bond in self.a2b[atom]:
                reverse_bond = self.b2revb[bond]
                if bond not in bonds_to_delete:
                    bonds_to_delete.append(bond)
                if reverse_bond not in bonds_to_delete:
                    bonds_to_delete.append(reverse_bond)

        # Do mask operation on atoms
        self.f_atoms = [
            self.f_atoms[idx] if idx not in atom_idx_to_remove else mask_atom_features(self.f_atoms[idx],
                                                                                       overwrite_default_atom=self.overwrite_default_atom_features)
            for idx in range(len(self.f_atoms))]

        bond_index_to_be_masked = [i for i, val in enumerate(self.b2a) if val in atom_idx_to_remove]

        self.f_bonds = [
            self.f_bonds[idx] if idx not in bond_index_to_be_masked else
            mask_bond_atom_features(self.f_bonds[idx], overwrite_default_atom=self.overwrite_default_atom_features,
                                    overwrite_default_bond=self.overwrite_default_bond_features) for idx in
            range(len(self.f_bonds))]

        # Do deletion operation on bonds

        masked_directional_bond_index_flattened = bonds_to_delete

        # mask the bond feature, this operation is important, since if we directly delete the bond features then all the bond index will need to be updated
        # recursively. Although it may not be that memory efficient but it can be more computational efficient

        self.f_bonds = [
            self.f_bonds[idx] if idx not in masked_directional_bond_index_flattened
            else mask_bond_features(self.f_bonds[idx], overwrite_default_atom=self.overwrite_default_atom_features,
                                    overwrite_default_bond=self.overwrite_default_bond_features) for idx in
            range(len(self.f_bonds))]

        # it will delete the bonds, and the bonds index will be replaced with 0 in batch graph construction
        self.a2b = [[bond for bond in bonds if bond not in masked_directional_bond_index_flattened] for bonds in
                    self.a2b]

        # for the bond which is deleted it will originates from an empty atom, a special token del is places first, then it will be replaced with 0 in batch graph construction
        self.b2a = ['del' if index in masked_directional_bond_index_flattened else value for index, value in
                    enumerate(self.b2a)]

        # for the bond which is deleted it will originates from an empty atom, a special token del is places first, then it will be replaced with 0 in batch graph construction
        self.b2revb = ['del' if value in masked_directional_bond_index_flattened else value for value in self.b2revb]

        self.final_subgraph_masked_atom_index = atom_idx_to_remove

        self.final_subgraph_deleted_bond_index = bonds_to_delete


class BatchMolGraphPretrain:
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of pretraining augmentated molecules.

    Note: Although a mixture of MolGraphPretrain objects with different augmentations can be used to form one batch graph.
    For now, the best practice is to form different batch graphs for different augmentations.

    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    * :code:`b2br`: (Optional): A mapping from f_bonds to real bonds in molecule recorded in targets.
    """

    def __init__(self, mol_graphs_pretrain: List[MolGraphPretrain]):
        r"""
        :param mol_graphs_pretrain: A list of :class:`MolGraphPretrain`\ s from which to construct the :class:`BatchMolGraphPretrain`.
        """
        self.mol_graphs_pretrain = mol_graphs_pretrain
        self.overwrite_default_atom_features = mol_graphs_pretrain[0].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs_pretrain[0].overwrite_default_bond_features
        self.is_reaction = mol_graphs_pretrain[0].is_reaction
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
        # The zero padding become very important for pretraining since the deleted connections will refer to it
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        final_masked_atom_index = []  # The index for all the masked atoms (update to batch version)
        masked_atom_label_list = []  # The label for all the masked atoms (update to batch version)
        for mol_graph_pretrain in mol_graphs_pretrain:
            f_atoms.extend(mol_graph_pretrain.f_atoms)
            f_bonds.extend(mol_graph_pretrain.f_bonds)

            if mol_graph_pretrain.final_masked_atom_index is not None:
                for m in range(len(mol_graph_pretrain.final_masked_atom_index)):
                    final_masked_atom_index.append(self.n_atoms + mol_graph_pretrain.final_masked_atom_index[m])

                masked_atom_label_list.extend(mol_graph_pretrain.masked_atom_label_list)

            for a in range(mol_graph_pretrain.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph_pretrain.a2b[a]])
                # For the deleted bond connected atom, it will be padded with 0s by self.a2b = torch.tensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)], dtype=torch.long)

            for b in range(mol_graph_pretrain.n_bonds):
                # There are 'del' in b2a and b2revb which can not add with an int
                if mol_graph_pretrain.b2a[b] != 'del':
                    b2a.append(self.n_atoms + mol_graph_pretrain.b2a[b])
                else:
                    b2a.append('del')

                if mol_graph_pretrain.b2revb[b] != 'del':
                    b2revb.append(self.n_bonds + mol_graph_pretrain.b2revb[b])
                else:
                    b2revb.append('del')

            self.a_scope.append((self.n_atoms, mol_graph_pretrain.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph_pretrain.n_bonds))
            self.n_atoms += mol_graph_pretrain.n_atoms
            self.n_bonds += mol_graph_pretrain.n_bonds

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.tensor(f_atoms, dtype=torch.float)
        self.f_bonds = torch.tensor(f_bonds, dtype=torch.float)
        self.a2b = torch.tensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)],
                                dtype=torch.long)
        # Replace 'del' with 0 in b2a
        b2a_replaced = [0 if x == 'del' else x for x in b2a]
        # Replace 'del' with 0 in b2revb
        b2revb_replaced = [0 if x == 'del' else x for x in b2revb]
        self.b2a = torch.tensor(b2a_replaced, dtype=torch.long)
        self.b2revb = torch.tensor(b2revb_replaced, dtype=torch.long)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages
        self.b2br = None  # only needed in predictions of atomic/bond targets
        self.final_masked_atom_index = final_masked_atom_index
        self.masked_atom_label_list = masked_atom_label_list

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
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
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages=atom_messages,
                                                     overwrite_default_atom=self.overwrite_default_atom_features,
                                                     overwrite_default_bond=self.overwrite_default_bond_features):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.Tensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.
        Holds true as all the deleted connections will be refer to zero paddings

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.Tensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.
        This one holds true after the atom masking, bond deletion as well as the subgraph deletion operation.
        If the connection is deleted then neighbor will refer to the 0 paddings

        :return: A PyTorch tensor containing the mapping from each atom index to all the neighboring atom indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1

            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a

    def get_b2br(self) -> torch.Tensor:
        """
        Computes (if necessary) and returns a mapping from f_bonds to real bonds in molecule recorded in targets.
        This holds true as well. Mainly will refer to the f_bonds. The deleted bonds will be all 0s

        :return: A PyTorch tensor containing the mapping from f_bonds to real bonds in molecule recorded in targets.
        """
        if self.b2br is None:
            n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
            b2br = []
            for mol_graph in self.mol_graphs:
                b2br.append(mol_graph.b2br + n_bonds)
                n_bonds += mol_graph.n_bonds
            b2br = np.concatenate(b2br, axis=0)
            self.b2br = torch.tensor(b2br, dtype=torch.long)

        return self.b2br


def mol2graphMaskAtom(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
                      atom_features_batch: List[np.array] = (None,),
                      bond_features_batch: List[np.array] = (None,),
                      overwrite_default_atom_features: bool = False,
                      overwrite_default_bond_features: bool = False,
                      mask_atom_pre_auto: bool = True,
                      mask_atom_pre_percent: float = 0.0,
                      mask_atom_pre_extra_mask_idx_batch: List[List[Union[int, float]]] = None,
                      ) -> BatchMolGraphPretrain:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param atom_features_batch: A list of 2D numpy array containing additional atom features to featurize the molecule.
    :param bond_features_batch: A list of 2D numpy array containing additional bond features to featurize the molecule.
    :param overwrite_default_atom_features: Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating.
    :param overwrite_default_bond_features: Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating.
    :param mask_atom_pre_auto: Boolean to state whether the mask atom pretrain is done by auto masking with random.
    :param mask_atom_pre_percent: Float to state the percentage of atoms to mask in mask atom pretrain.
    :param mask_atom_pre_extra_mask_idx_batch: A list of lists of int or float to state pre_determined masked atom index.
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.

    """
    MolGraphPretrain_MA_list = [MolGraphPretrain(mol, af, bf,
                                                 overwrite_default_atom_features=overwrite_default_atom_features,
                                                 overwrite_default_bond_features=overwrite_default_bond_features,
                                                 mask_atom_pre_auto=mask_atom_pre_auto,
                                                 mask_atom_pre_percent=mask_atom_pre_percent)
                                for mol, af, bf, ma_idx
                                in zip_longest(mols, atom_features_batch, bond_features_batch,
                                               mask_atom_pre_extra_mask_idx_batch)]

    for molgraphpre in MolGraphPretrain_MA_list:
        molgraphpre.masked_atom_pretraining()

    BatchMolGraphPretrain_MA = BatchMolGraphPretrain(MolGraphPretrain_MA_list)

    return BatchMolGraphPretrain_MA


def mol2graphBondDele(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
                      atom_features_batch: List[np.array] = (None,),
                      bond_features_batch: List[np.array] = (None,),
                      overwrite_default_atom_features: bool = False,
                      overwrite_default_bond_features: bool = False,
                      mask_bond_pre_auto: bool = True,
                      mask_bond_pre_percent: float = 0.0,
                      mask_bond_pre_extra_mask_idx_batch: List[List[Union[int, float]]] = None,
                      ) -> BatchMolGraphPretrain:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param atom_features_batch: A list of 2D numpy array containing additional atom features to featurize the molecule.
    :param bond_features_batch: A list of 2D numpy array containing additional bond features to featurize the molecule.
    :param overwrite_default_atom_features: Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating.
    :param overwrite_default_bond_features: Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating.
    :param mask_bond_pre_auto: Boolean to state whether the bond deletion  is done by auto delete with random.
    :param mask_bond_pre_percent: Float to state the percentage of bonds to mask in bond deletion pretrain.
    :param mask_bond_pre_extra_mask_idx: A list of lists of int or float to state pre_determined masked bond index.
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.

    """
    MolGraphPretrain_BD_list = [MolGraphPretrain(mol, af, bf,
                                                 overwrite_default_atom_features=overwrite_default_atom_features,
                                                 overwrite_default_bond_features=overwrite_default_bond_features,
                                                 mask_bond_pre_auto=mask_bond_pre_auto,
                                                 mask_bond_pre_percent=mask_bond_pre_percent)
                                for mol, af, bf, mb_idx
                                in zip_longest(mols, atom_features_batch, bond_features_batch,
                                               mask_bond_pre_extra_mask_idx_batch)]

    for molgraphpre in MolGraphPretrain_BD_list:
        molgraphpre.bond_deletion_complete()

    BatchMolGraphPretrain_BD = BatchMolGraphPretrain(MolGraphPretrain_BD_list)

    return BatchMolGraphPretrain_BD


def mol2graphSubgraphDele(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
                          atom_features_batch: List[np.array] = (None,),
                          bond_features_batch: List[np.array] = (None,),
                          overwrite_default_atom_features: bool = False,
                          overwrite_default_bond_features: bool = False,
                          mask_subgraph_pre_percent: float = 0.0,
                          center_list: List[int] = None,
                          ) -> BatchMolGraphPretrain:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param atom_features_batch: A list of 2D numpy array containing additional atom features to featurize the molecule.
    :param bond_features_batch: A list of 2D numpy array containing additional bond features to featurize the molecule.
    :param overwrite_default_atom_features: Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating.
    :param overwrite_default_bond_features: Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating.
    :param mask_subgraph_pre_percent: Float to state the percentage of subgraph to mask in subgraph deletion pretrain.
    :param center_list: A list of int to state the start center atom in each molecule of subgraph to mask in subgraph deletion pretrain.
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.

    """
    MolGraphPretrain_SG_list = [MolGraphPretrain(mol, af, bf,
                                                 overwrite_default_atom_features=overwrite_default_atom_features,
                                                 overwrite_default_bond_features=overwrite_default_bond_features,
                                                 mask_subgraph_pre_percent=mask_subgraph_pre_percent, )
                                for mol, af, bf
                                in zip_longest(mols, atom_features_batch, bond_features_batch)]

    for i, molgraphpre in enumerate(MolGraphPretrain_SG_list):
        molgraphpre.subgraph_deletion(center_list[i])

    BatchMolGraphPretrain_SG = BatchMolGraphPretrain(MolGraphPretrain_SG_list)

    return BatchMolGraphPretrain_SG
