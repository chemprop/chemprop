import threading
from collections import OrderedDict
from random import Random
from typing import Dict, Iterator, List, Optional, Union, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem

from .scaler import StandardScaler, AtomBondScaler
from chemprop.features import get_features_generator
from chemprop.features import BatchMolGraph, MolGraph
from chemprop.features import is_explicit_h, is_reaction, is_adding_hs, is_mol, is_keeping_atom_map
from chemprop.rdkit import make_mol

# Cache of graph featurizations
CACHE_GRAPH = True
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}


# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]] = {}


def cache_graph() -> bool:
    r"""Returns whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    return CACHE_GRAPH


def set_cache_graph(cache_graph: bool) -> None:
    r"""Sets whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph


def empty_cache():
    r"""Empties the cache of :class:`~chemprop.features.MolGraph` and RDKit molecules."""
    SMILES_TO_GRAPH.clear()
    SMILES_TO_MOL.clear()


def cache_mol() -> bool:
    r"""Returns whether RDKit molecules will be cached."""
    return CACHE_MOL


def set_cache_mol(cache_mol: bool) -> None:
    r"""Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol


class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    def __init__(self,
                 smiles: List[str],
                 targets: List[Optional[float]] = None,
                 atom_targets: List[Optional[float]] = None,
                 bond_targets: List[Optional[float]] = None,
                 row: OrderedDict = None,
                 data_weight: float = None,
                 gt_targets: List[List[bool]] = None,
                 lt_targets: List[List[bool]] = None,
                 features: np.ndarray = None,
                 features_generator: List[str] = None,
                 phase_features: List[float] = None,
                 atom_features: np.ndarray = None,
                 atom_descriptors: np.ndarray = None,
                 bond_features: np.ndarray = None,
                 bond_descriptors: np.ndarray = None,
                 raw_constraints: np.ndarray = None,
                 constraints: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False):
        """
        :param smiles: A list of the SMILES strings for the molecules.
        :param targets: A list of targets for the molecule (contains None for unknown target values).
        :param atom_targets: A list of targets for the atomic properties.
        :param bond_targets: A list of targets for the bond properties.
        :param row: The raw CSV row containing the information for this molecule.
        :param data_weight: Weighting of the datapoint for the loss function.
        :param gt_targets: Indicates whether the targets are an inequality regression target of the form ">x".
        :param lt_targets: Indicates whether the targets are an inequality regression target of the form "<x".
        :param features: A numpy array containing additional features (e.g., Morgan fingerprint).
        :param features_generator: A list of features generators to use.
        :param phase_features: A one-hot vector indicating the phase of the data, as used in spectra data.
        :param atom_descriptors: A numpy array containing additional atom descriptors to featurize the molecule.
        :param bond_descriptors: A numpy array containing additional bond descriptors to featurize the molecule.
        :param raw_constraints: A numpy array containing all user-provided atom/bond-level constraints in input data.
        :param constraints: A numpy array containing atom/bond-level constraints that are used in training. Param constraints is a subset of param raw_constraints.
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features.
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features.

        """
        self.smiles = smiles
        self.targets = targets
        self.atom_targets = atom_targets
        self.bond_targets = bond_targets
        self.row = row
        self.features = features
        self.features_generator = features_generator
        self.phase_features = phase_features
        self.atom_descriptors = atom_descriptors
        self.bond_descriptors = bond_descriptors
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.constraints = constraints
        self.raw_constraints = raw_constraints
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features
        self.is_mol_list = [is_mol(s) for s in smiles]
        self.is_reaction_list = [is_reaction(x) for x in self.is_mol_list]
        self.is_explicit_h_list = [is_explicit_h(x) for x in self.is_mol_list]
        self.is_adding_hs_list = [is_adding_hs(x) for x in self.is_mol_list]
        self.is_keeping_atom_map_list = [is_keeping_atom_map(x) for x in self.is_mol_list]

        if data_weight is not None:
            self.data_weight = data_weight
        if gt_targets is not None:
            self.gt_targets = gt_targets
        if lt_targets is not None:
            self.lt_targets = lt_targets

        # Generate additional features if given a generator
        if self.features_generator is not None:
            if self.features is None:
                self.features = []
            else:
                self.features = list(self.features)

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                for m, reaction in zip(self.mol, self.is_reaction_list):
                    if not reaction:
                        if m is not None and m.GetNumHeavyAtoms() > 0:
                            self.features.extend(features_generator(m))
                        # for H2
                        elif m is not None and m.GetNumHeavyAtoms() == 0:
                            # not all features are equally long, so use methane as dummy molecule to determine length
                            self.features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))
                    else:
                        if m[0] is not None and m[1] is not None and m[0].GetNumHeavyAtoms() > 0:
                            self.features.extend(features_generator(m[0]))
                        elif m[0] is not None and m[1] is not None and m[0].GetNumHeavyAtoms() == 0:
                            self.features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))   
                    

            self.features = np.array(self.features)

        # Fix nans in features
        replace_token = 0
        if self.features is not None:
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Fix nans in atom_descriptors
        if self.atom_descriptors is not None:
            self.atom_descriptors = np.where(np.isnan(self.atom_descriptors), replace_token, self.atom_descriptors)

        # Fix nans in atom_features
        if self.atom_features is not None:
            self.atom_features = np.where(np.isnan(self.atom_features), replace_token, self.atom_features)

        # Fix nans in bond_descriptors
        if self.bond_descriptors is not None:
            self.bond_descriptors = np.where(np.isnan(self.bond_descriptors), replace_token, self.bond_descriptors)

        # Fix nans in bond_features
        if self.bond_features is not None:
            self.bond_features = np.where(np.isnan(self.bond_features), replace_token, self.bond_features)

        # Save a copy of the raw features and targets to enable different scaling later on
        self.raw_features, self.raw_targets, self.raw_atom_targets, self.raw_bond_targets = \
            self.features, self.targets, self.atom_targets, self.bond_targets
        self.raw_atom_descriptors, self.raw_atom_features, self.raw_bond_descriptors, self.raw_bond_features = \
            self.atom_descriptors, self.atom_features, self.bond_descriptors, self.bond_features

    @property
    def mol(self) -> List[Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]]:
        """Gets the corresponding list of RDKit molecules for the corresponding SMILES list."""
        if self.atom_targets is not None or self.bond_targets is not None:
            # When the original atom mapping is used, the explicit hydrogens specified in the input SMILES should be used
            # However, the explicit Hs can only be added for reactions with `--explicit_h` flag
            # To fix this, the attribute of `keep_h_list` in make_mols() is set to match the `keep_atom_map_list`
            mol = make_mols(smiles=self.smiles,
                            reaction_list=self.is_reaction_list,
                            keep_h_list=self.is_keeping_atom_map_list,
                            add_h_list=self.is_adding_hs_list,
                            keep_atom_map_list=self.is_keeping_atom_map_list)
        else:
            mol = make_mols(smiles=self.smiles,
                            reaction_list=self.is_reaction_list,
                            keep_h_list=self.is_explicit_h_list,
                            add_h_list=self.is_adding_hs_list,
                            keep_atom_map_list=self.is_keeping_atom_map_list)
        if cache_mol():
            for s, m in zip(self.smiles, mol):
                SMILES_TO_MOL[s] = m

        return mol

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in the :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return len(self.smiles)

    @property
    def number_of_atoms(self) -> int:
        """
        Gets the number of atoms in the :class:`MoleculeDatapoint`.

        :return: A list of number of atoms for each molecule.
        """
        return [len(self.mol[i].GetAtoms()) for i in range(self.number_of_molecules)]

    @property
    def number_of_bonds(self) -> List[int]:
        """
        Gets the number of bonds in the :class:`MoleculeDatapoint`.

        :return: A list of number of bonds for each molecule.
        """
        return [len(self.mol[i].GetBonds()) for i in range(self.number_of_molecules)]

    @property
    def bond_types(self) -> List[List[float]]:
        """
        Gets the bond types in the :class:`MoleculeDatapoint`.

        :return: A list of bond types for each molecule.
        """
        return [[b.GetBondTypeAsDouble() for b in self.mol[i].GetBonds()] for i in range(self.number_of_molecules)]
    @property
    def max_molwt(self) -> float:
        """
        Gets the maximum molecular weight among all the molecules in the :class:`MoleculeDatapoint`.

        :return: The maximum molecular weight.
        """
        return max(Chem.rdMolDescriptors.CalcExactMolWt(mol) for mol in self.mol)

    def set_features(self, features: np.ndarray) -> None:
        """
        Sets the features of the molecule.

        :param features: A 1D numpy array of features for the molecule.
        """
        self.features = features

    def set_atom_descriptors(self, atom_descriptors: np.ndarray) -> None:
        """
        Sets the atom descriptors of the molecule.

        :param atom_descriptors: A 1D numpy array of atom descriptors for the molecule.
        """
        self.atom_descriptors = atom_descriptors

    def set_atom_features(self, atom_features: np.ndarray) -> None:
        """
        Sets the atom features of the molecule.

        :param atom_features: A 1D numpy array of atom features for the molecule.
        """
        self.atom_features = atom_features

    def set_bond_descriptors(self, bond_descriptors: np.ndarray) -> None:
        """
        Sets the atom descriptors of the molecule.

        :param bond_descriptors: A 1D numpy array of bond descriptors for the molecule.
        """
        self.bond_descriptors = bond_descriptors

    def set_bond_features(self, bond_features: np.ndarray) -> None:
        """
        Sets the bond features of the molecule.

        :param bond_features: A 1D numpy array of bond features for the molecule.
        """
        self.bond_features = bond_features

    def extend_features(self, features: np.ndarray) -> None:
        """
        Extends the features of the molecule.

        :param features: A 1D numpy array of extra features for the molecule.
        """
        self.features = np.append(self.features, features) if self.features is not None else features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        self.features, self.targets, self.atom_targets, self.bond_targets = \
            self.raw_features, self.raw_targets, self.raw_atom_targets, self.raw_bond_targets
        self.atom_descriptors, self.atom_features, self.bond_descriptors, self.bond_features = \
            self.raw_atom_descriptors, self.raw_atom_features, self.raw_bond_descriptors, self.raw_bond_features


class MoleculeDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to their attributes."""

    def __init__(self, data: List[MoleculeDatapoint]):
        r"""
        :param data: A list of :class:`MoleculeDatapoint`\ s.
        """
        self._data = data
        self._batch_graph = None
        self._random = Random()

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Returns a list containing the SMILES list associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
        """
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]

        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]], List[Tuple[Chem.Mol, Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]]]:
        """
        Returns a list of the RDKit molecules associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned RDKit molecules to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of RDKit molecules, depending on :code:`flatten`.
        """
        if flatten:
            return [mol for d in self._data for mol in d.mol]

        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in each :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    @property
    def number_of_atoms(self) -> List[List[int]]:
        """
        Gets the number of atoms in each :class:`MoleculeDatapoint`.

        :return: A list of number of atoms for each molecule.
        """
        return [d.number_of_atoms for d in self._data]

    @property
    def number_of_bonds(self) -> List[List[int]]:
        """
        Gets the number of bonds in each :class:`MoleculeDatapoint`.

        :return: A list of number of bonds for each molecule.
        """
        return [d.number_of_bonds for d in self._data]

    @property
    def bond_types(self) -> List[List[float]]:
        """
        Gets the bond types in each :class:`MoleculeDatapoint`.

        :return: A list of bond types for each molecule.
        """
        return [d.bond_types for d in self._data]

    @property
    def is_atom_bond_targets(self) -> bool:
        """
        Gets the Boolean whether this is atomic/bond properties prediction.

        :return: A Boolean value.
        """
        if self._data[0].atom_targets is None and self._data[0].bond_targets is None:
            return False
        else:
            return True

    def batch_graph(self) -> List[BatchMolGraph]:
        r"""
        Constructs a :class:`~chemprop.features.BatchMolGraph` with the graph featurization of all the molecules.

        .. note::
           The :class:`~chemprop.features.BatchMolGraph` is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to :meth:`batch_graph`. This means that if the underlying
           set of :class:`MoleculeDatapoint`\ s changes, then the returned :class:`~chemprop.features.BatchMolGraph`
           will be incorrect for the underlying data.

        :return: A list of :class:`~chemprop.features.BatchMolGraph` containing the graph featurization of all the
                 molecules in each :class:`MoleculeDatapoint`.
        """
        if self._batch_graph is None:
            self._batch_graph = []

            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for s, m in zip(d.smiles, d.mol):
                    if s in SMILES_TO_GRAPH:
                        mol_graph = SMILES_TO_GRAPH[s]
                    else:
                        if len(d.smiles) > 1 and (d.atom_features is not None or d.bond_features is not None):
                            raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                                      'per input (i.e., number_of_molecules = 1).')

                        mol_graph = MolGraph(m, d.atom_features, d.bond_features,
                                             overwrite_default_atom_features=d.overwrite_default_atom_features,
                                             overwrite_default_bond_features=d.overwrite_default_bond_features)
                        if cache_graph():
                            SMILES_TO_GRAPH[s] = mol_graph
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)

            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]

        return self._batch_graph

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def phase_features(self) -> List[np.ndarray]:
        """
        Returns the phase features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the phase features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].phase_features is None:
            return None

        return [d.phase_features for d in self._data]

    def atom_features(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_features is None:
            return None

        return [d.atom_features for d in self._data]

    def atom_descriptors(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_descriptors is None:
            return None

        return [d.atom_descriptors for d in self._data]

    def bond_features(self) -> List[np.ndarray]:
        """
        Returns the bond features associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the bond features
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].bond_features is None:
            return None

        return [d.bond_features for d in self._data]

    def bond_descriptors(self) -> List[np.ndarray]:
        """
        Returns the bond descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the bond descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].bond_descriptors is None:
            return None

        return [d.bond_descriptors for d in self._data]

    def constraints(self) -> List[np.ndarray]:
        """
        Return the constraints applied in atomic/bond properties prediction.
        """
        constraints = []
        for d in self._data:
            if d.constraints is None :
                natom_targets = len(d.atom_targets) if d.atom_targets is not None else 0
                nbond_targets = len(d.bond_targets) if d.bond_targets is not None else 0
                ntargets = natom_targets + nbond_targets
                constraints.append([None] * ntargets)
            else:
                constraints.append(d.constraints)
        return constraints

    def data_weights(self) -> List[float]:
        """
        Returns the loss weighting associated with each datapoint.
        """
        if not hasattr(self._data[0], 'data_weight'):
            return [1. for d in self._data]

        return [d.data_weight for d in self._data]

    def atom_bond_data_weights(self) -> List[List[float]]:
        """
        Returns the loss weighting associated with each datapoint for atomic/bond properties prediction.
        """
        targets = self.targets()
        data_weights = self.data_weights()
        atom_bond_data_weights = [[] for _ in targets[0]]
        for i, tb in enumerate(targets):
            weight = data_weights[i]
            for j, x in enumerate(tb): 
                atom_bond_data_weights[j] += [1. * weight] * len(x)

        return atom_bond_data_weights

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.targets for d in self._data]
    
    def mask(self) -> List[List[bool]]:
        """
        Returns whether the targets associated with each molecule and task are present.

        :return: A list of list of booleans associated with targets.
        """
        targets = self.targets()
        if self.is_atom_bond_targets:
            mask = []
            for dt in zip(*targets):
                dt = np.concatenate(dt)
                mask.append([x is not None for x in dt])
        else:
            mask = [[t is not None for t in dt] for dt in targets]
            mask = list(zip(*mask))
        return mask

    def gt_targets(self) -> List[np.ndarray]:
        """
        Returns indications of whether the targets associated with each molecule are greater-than inequalities.
        
        :return: A list of lists of booleans indicating whether the targets in those positions are greater-than inequality targets.
        """
        if not hasattr(self._data[0], 'gt_targets'):
            return None

        return [d.gt_targets for d in self._data]

    def lt_targets(self) -> List[np.ndarray]:
        """
        Returns indications of whether the targets associated with each molecule are less-than inequalities.
        
        :return: A list of lists of booleans indicating whether the targets in those positions are less-than inequality targets.
        """
        if not hasattr(self._data[0], 'lt_targets'):
            return None

        return [d.lt_targets for d in self._data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the additional features vector associated with the molecules.

        :return: The size of the additional features vector.
        """
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def atom_descriptors_size(self) -> int:
        """
        Returns the size of custom additional atom descriptors vector associated with the molecules.

        :return: The size of the additional atom descriptor vector.
        """
        return len(self._data[0].atom_descriptors[0]) \
            if len(self._data) > 0 and self._data[0].atom_descriptors is not None else None

    def atom_features_size(self) -> int:
        """
        Returns the size of custom additional atom features vector associated with the molecules.

        :return: The size of the additional atom feature vector.
        """
        return len(self._data[0].atom_features[0]) \
            if len(self._data) > 0 and self._data[0].atom_features is not None else None

    def bond_descriptors_size(self) -> int:
        """
        Returns the size of custom additional bond descriptors vector associated with the molecules.

        :return: The size of the additional bond descriptor vector.
        """
        return len(self._data[0].bond_descriptors[0]) \
            if len(self._data) > 0 and self._data[0].bond_descriptors is not None else None

    def bond_features_size(self) -> int:
        """
        Returns the size of custom additional bond features vector associated with the molecules.

        :return: The size of the additional bond feature vector.
        """
        return len(self._data[0].bond_features[0]) \
            if len(self._data) > 0 and self._data[0].bond_features is not None else None

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0,
                           scale_atom_descriptors: bool = False, scale_bond_descriptors: bool = False) -> StandardScaler:
        """
        Normalizes the features of the dataset using a :class:`~chemprop.data.StandardScaler`.

        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each feature independently.

        If a :class:`~chemprop.data.StandardScaler` is provided, it is used to perform the normalization.
        Otherwise, a :class:`~chemprop.data.StandardScaler` is first fit to the features in this dataset
        and is then used to perform the normalization.

        :param scaler: A fitted :class:`~chemprop.data.StandardScaler`. If it is provided it is used,
                       otherwise a new :class:`~chemprop.data.StandardScaler` is first fitted to this
                       data and is then used.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        :param scale_atom_descriptors: If the features that need to be scaled are atom features rather than molecule.
        :param scale_bond_descriptors: If the features that need to be scaled are bond features rather than molecule.
        :return: A fitted :class:`~chemprop.data.StandardScaler`. If a :class:`~chemprop.data.StandardScaler`
                 is provided as a parameter, this is the same :class:`~chemprop.data.StandardScaler`. Otherwise,
                 this is a new :class:`~chemprop.data.StandardScaler` that has been fit on this dataset.
        """
        if len(self._data) == 0 or \
                (self._data[0].features is None and not scale_bond_descriptors and not scale_atom_descriptors):
            return None

        if scaler is None:
            if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
                features = np.vstack([d.raw_atom_descriptors for d in self._data])
            elif scale_atom_descriptors and not self._data[0].atom_features is None:
                features = np.vstack([d.raw_atom_features for d in self._data])
            elif scale_bond_descriptors and not self._data[0].bond_descriptors is None:
                features = np.vstack([d.raw_bond_descriptors for d in self._data])
            elif scale_bond_descriptors and not self._data[0].bond_features is None:
                features = np.vstack([d.raw_bond_features for d in self._data])
            else:
                features = np.vstack([d.raw_features for d in self._data])
            scaler = StandardScaler(replace_nan_token=replace_nan_token)
            scaler.fit(features)

        if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
            for d in self._data:
                d.set_atom_descriptors(scaler.transform(d.raw_atom_descriptors))
        elif scale_atom_descriptors and not self._data[0].atom_features is None:
            for d in self._data:
                d.set_atom_features(scaler.transform(d.raw_atom_features))
        elif scale_bond_descriptors and not self._data[0].bond_descriptors is None:
            for d in self._data:
                d.set_bond_descriptors(scaler.transform(d.raw_bond_descriptors))
        elif scale_bond_descriptors and not self._data[0].bond_features is None:
            for d in self._data:
                d.set_bond_features(scaler.transform(d.raw_bond_features))
        else:
            for d in self._data:
                d.set_features(scaler.transform(d.raw_features.reshape(1, -1))[0])

        return scaler

    def normalize_targets(self) -> StandardScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.
        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.
        This should only be used for regression datasets.
        :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
        """
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

        return scaler

    def normalize_atom_bond_targets(self) -> AtomBondScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.AtomBondScaler`.

        The :class:`~chemprop.data.AtomBondScaler` subtracts the mean and divides by the standard deviation
        for each task independently.

        This should only be used for regression datasets.

        :return: A :class:`~chemprop.data.AtomBondScaler` fitted to the targets.
        """
        atom_targets = self._data[0].atom_targets
        bond_targets = self._data[0].bond_targets
        n_atom_targets = len(atom_targets) if atom_targets is not None else 0
        n_bond_targets = len(bond_targets) if bond_targets is not None else 0
        n_atoms, n_bonds = self.number_of_atoms, self.number_of_bonds

        targets = [d.raw_targets for d in self._data]
        targets = [np.concatenate(x).reshape([-1, 1]) for x in zip(*targets)]
        scaler = AtomBondScaler(
            n_atom_targets=n_atom_targets,
            n_bond_targets=n_bond_targets,
        ).fit(targets)
        scaled_targets = scaler.transform(targets)
        for i in range(n_atom_targets):
            scaled_targets[i] = np.split(np.array(scaled_targets[i]).flatten(), np.cumsum(np.array(n_atoms)))[:-1]
        for i in range(n_bond_targets):
            scaled_targets[i+n_atom_targets] = np.split(np.array(scaled_targets[i+n_atom_targets]).flatten(), np.cumsum(np.array(n_bonds)))[:-1]
        scaled_targets = np.array(scaled_targets, dtype=object).T
        self.set_targets(scaled_targets)

        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        if not len(self._data) == len(targets):
            raise ValueError(
                "number of molecules and targets must be of same length! "
                f"num molecules: {len(self._data)}, num targets: {len(targets)}"
            )
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]


class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([any(target == 1 for target in datapoint.targets) for datapoint in dataset])

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    r"""
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.

    Additionally, precomputes the :class:`~chemprop.features.BatchMolGraph` for the constructed
    :class:`MoleculeDataset`.

    :param data: A list of :class:`MoleculeDatapoint`\ s.
    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\ s.
    """
    data = MoleculeDataset(data)
    data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

    return data


class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def gt_targets(self) -> List[List[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')
        
        if not hasattr(self._dataset[0],'gt_targets'):
            return None

        return [self._dataset[index].gt_targets for index in self._sampler]

    @property
    def lt_targets(self) -> List[List[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        if not hasattr(self._dataset[0],'lt_targets'):
            return None

        return [self._dataset[index].lt_targets for index in self._sampler]


    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()

    
def make_mols(smiles: List[str], reaction_list: List[bool], keep_h_list: List[bool], add_h_list: List[bool], keep_atom_map_list: List[bool]):
    """
    Builds a list of RDKit molecules (or a list of tuples of molecules if reaction is True) for a list of smiles.

    :param smiles: List of SMILES strings.
    :param reaction_list: List of booleans whether the SMILES strings are to be treated as a reaction.
    :param keep_h_list: List of booleans whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h_list: List of booleasn whether to add hydrogens to the input smiles.
    :param keep_atom_map_list: List of booleasn whether to keep the original atom mapping.
    :return: List of RDKit molecules or list of tuple of molecules.
    """
    mol = []
    for s, reaction, keep_h, add_h, keep_atom_map in zip(smiles, reaction_list, keep_h_list, add_h_list, keep_atom_map_list):
        if reaction:
            mol.append(SMILES_TO_MOL[s] if s in SMILES_TO_MOL else (make_mol(s.split(">")[0], keep_h, add_h, keep_atom_map), make_mol(s.split(">")[-1], keep_h, add_h, keep_atom_map)))
        else:
            mol.append(SMILES_TO_MOL[s] if s in SMILES_TO_MOL else make_mol(s, keep_h, add_h, keep_atom_map))
    return mol

