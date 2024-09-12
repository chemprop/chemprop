from typing import NamedTuple, Optional
from collections import deque
import numpy as np

import torch
from dataclasses import dataclass, field

class MolGraph(NamedTuple):
    """A :class:`MolGraph` represents the graph featurization of a molecule."""

    V: np.ndarray
    """an array of shape ``V x d_v`` containing the atom features of the molecule"""
    E: np.ndarray
    """an array of shape ``E x d_e`` containing the bond features of the molecule"""
    edge_index: np.ndarray
    """an array of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: np.ndarray
    """A array of shape ``E`` that maps from an edge index to the index of the source of the reverse edge in :attr:`edge_index` attribute."""


@dataclass
class MolGraphPretrain:
    """for this object the main methods such as mask and bond deletion will modify the attribute irrevisble so recommend one modification for one object"""
    V: np.ndarray
    """An array of shape ``V x d_v`` containing the atom features of the molecule"""

    E: np.ndarray
    """An array of shape ``E x d_e`` containing the bond features of the molecule"""

    edge_index: np.ndarray
    """An array of shape ``2 x E`` containing the edges of the graph in COO format"""
    """For bond deletion and subgraph deletion, the shape will change into 2 x E-deleted bonds"""

    rev_edge_index: np.ndarray
    """An array of shape ``E`` that maps from an edge index to the index of the source of the reverse edge in :attr:`edge_index` attribute."""

    mask_atom_pre_percent: Optional[float] = field(default=None)
    """Float to state the percentage of atoms to mask in mask atom pretrain."""

    mask_bond_pre_percent: Optional[float] = field(default=None)
    """Float to state the percentage of bonds to mask in bond deletion pretrain."""

    mask_subgraph_pre_percent: Optional[float] = field(default=None)
    """Float to state the percentage of subgraph to mask in subgraph deletion pretrain."""

    masked_atom_label_list: Optional[np.ndarray] = field(default=None)
    """The final masked atom num label used for masking atom pretraining"""

    masked_atom_index: Optional[np.ndarray] = field(default=None)
    """The final masked atom index used for masking atom pretraining"""

    atom_num_label: list = field(default_factory=list)
    """The atom num label used for masking atom pretraining"""

    masked_complete_bond_index: list = field(default_factory=list)
    """The final masked complete bond index used for bond deletion pretraining"""

    final_bond_masked_indices: list = field(default_factory=list)
    """The final masked bond index used for bond deletion pretraining"""

    subgraph_masked_atom_indices: list = field(default_factory=list)
    """The final masked atom index used for subgraph deletion pretraining"""

    subgraph_masked_bond_indices: list = field(default_factory=list)
    """The final masked bond index used for subgraph deletion pretraining"""

    def mask_atom_features(self, v: np.ndarray) -> np.ndarray:
        '''
        :param v: The original feature of atoms
        :return: a ndarray of 0, which replaces all the original feature of atoms
        '''
        atom_features_len = len(v)
        masked_atom_features = np.zeros(atom_features_len)
        return masked_atom_features

    def update_edge_index(self):
        """
        Update the edge_index after deletion to reflect the correct node indices.
        This is necessary because deleting edges might affect the node indices used in the edge_index.
        """
        unique_nodes = np.unique(self.edge_index)
        new_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_nodes)}

        # Update edge_index using the mapping
        for i in range(self.edge_index.shape[1]):
            self.edge_index[0, i] = new_index_map[self.edge_index[0, i]]
            self.edge_index[1, i] = new_index_map[self.edge_index[1, i]]

    def masked_atom_pretraining(self,mask_atom_pre_percent: float):
        '''
        The atom feature is masked by replacing original feature with 0
        The bond feature's atom part will be replaced with 0 if the atom the bond originates from is masked (now this is done automatically as the concate
        is done in message passing
        :return: No return. Will just modify MolGraphPretrain object into masked atom version.
        :Note: The MolGraphPretrain(object) will be modified, and for different pretraining operation should init a new object then use the operation function
        '''
        self.mask_atom_pre_percent = mask_atom_pre_percent

        masked_atom_index = torch.randperm(len(self.V))[
                                 :int(len(self.V) * self.mask_atom_pre_percent)].numpy()
        masked_atom_index = np.sort(masked_atom_index)
        self.masked_atom_index = masked_atom_index

        self.V = np.array([self.V[idx] if idx not in self.masked_atom_index else self.mask_atom_features(self.V[idx]) for
            idx in range(len(self.V))])

        self.masked_atom_label_list = np.array([
            self.atom_num_label[idx] for idx in self.masked_atom_index])


    def bond_deletion_complete(self, mask_bond_pre_percent: float):
        '''
        This operation is to delete certain percentage of the bond completely (i.e., we have directional bond, this will
        delete both the direction).

        :return:
        '''
        # Get random bond deletion index from number of complete bond
        self.mask_bond_pre_percent = mask_bond_pre_percent
        number_of_undirectional_bonds = len(self.E)//2
        masked_complete_bond_index = torch.randperm(number_of_undirectional_bonds)[
                                          :int(number_of_undirectional_bonds * self.mask_bond_pre_percent)]
        self.masked_complete_bond_index = masked_complete_bond_index.tolist()
        self.masked_complete_bond_index = sorted(self.masked_complete_bond_index)
        even_bond_indices = torch.arange(0, len(self.E), step=2)
        masked_even_indices = even_bond_indices[self.masked_complete_bond_index]
        masked_odd_indices = masked_even_indices + 1
        self.final_bond_masked_indices = torch.cat([masked_even_indices, masked_odd_indices]).tolist()
        # in v2 the connectivity is stored in edge index, as a result we can simply delete the connectivity by deleted both the list with the connections
        # Delete corresponding rows from edge_index and E
        # edge_index contains index of node delete edge won't change node order
        self.edge_index = np.delete(self.edge_index, self.final_bond_masked_indices, axis=1)
        self.E = np.delete(self.E, self.final_bond_masked_indices, axis=0)
        self.rev_edge_index = np.arange(len(self.E)).reshape(-1, 2)[:, ::-1].ravel()

    def subgraph_deletion(self, center: int, mask_subgraph_pre_percent: float):
        '''
        Remove a subgraph meaning mask the atom to be removed and remove all the bonds connected to it.

        :param center: The start atom to remove
        :param mask_subgraph_pre_percent: percent

        :return: it is just an operation on this object
        '''
        self.mask_subgraph_pre_percent = mask_subgraph_pre_percent
        # Step 1: BFS to identify all atoms in the subgraph
        num_of_atoms_of_removed_subgraph = int(np.floor(len(self.V) * self.mask_subgraph_pre_percent))
        atom_idx_to_remove = set()
        queue = deque([center])

        while queue and len(atom_idx_to_remove) < num_of_atoms_of_removed_subgraph:
            current_atom = queue.popleft()
            if current_atom not in atom_idx_to_remove:
                atom_idx_to_remove.add(current_atom)

                # Add all neighboring atoms to the queue
                for i in range(self.edge_index.shape[1]):
                    if self.edge_index[0, i] == current_atom and self.edge_index[1, i] not in atom_idx_to_remove:
                        queue.append(self.edge_index[1, i])
                    elif self.edge_index[1, i] == current_atom and self.edge_index[0, i] not in atom_idx_to_remove:
                        queue.append(self.edge_index[0, i])

        # Step 2: Identify bonds connected to the atoms to be removed
        self.subgraph_masked_bond_indices = [
            i for i in range(self.edge_index.shape[1])
            if self.edge_index[0, i] in atom_idx_to_remove or self.edge_index[1, i] in atom_idx_to_remove
        ]

        # Step 3: Remove bonds from edge_index and delete bond features in E
        self.edge_index = np.delete(self.edge_index, self.subgraph_masked_bond_indices, axis=1)
        self.E = np.delete(self.E, self.subgraph_masked_bond_indices, axis=0)

        # Update rev_edge_index
        self.rev_edge_index = np.arange(len(self.E)).reshape(-1, 2)[:, ::-1].ravel()

        # Store the removed atom indices
        self.subgraph_masked_atom_indices = list(atom_idx_to_remove)

        # Step 4: Mask the atom features in the subgraph
        for atom_idx in self.subgraph_masked_atom_indices:
            self.V[atom_idx] = self.mask_atom_features(self.V[atom_idx])

        # Store sorted information for reference or further operations
        self.subgraph_masked_atom_indices = sorted(self.subgraph_masked_atom_indices)
        self.subgraph_masked_bond_indices = sorted(self.subgraph_masked_bond_indices)