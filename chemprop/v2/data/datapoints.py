from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import InitVar, dataclass, field
from typing import List, Optional

import numpy as np
from rdkit import Chem

from chemprop.v2.utils import make_mol
from chemprop.featurizers import get_features_generator


@dataclass
class DatapointBase(ABC):
    """A `DatapointBase` is the base datapoint for both molecule- and reaction-type data"""

    targets: Optional[np.ndarray] = None
    row: OrderedDict = None
    data_weight: float = 1
    gt_targets: Optional[np.ndarray] = None
    lt_targets: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    features_generators: InitVar[Optional[List[str]]] = None
    phase_features: List[float] = None
    explicit_h: bool = False
    add_h: bool = False

    def __post_init__(self, features_generators):
        if self.features is not None and features_generators is not None:
            raise ValueError("Cannot provide both loaded features and features generators!")

        if features_generators is not None:
            self.features = self.generate_features(features_generators)

        replace_token = 0
        if self.features is not None:
            self.features[np.isnan(self.features)] = replace_token

        self._features = self.features
        self._targets = self.targets

    @property
    def num_tasks(self) -> int:
        return len(self.targets)

    @property
    @abstractmethod
    def number_of_molecules(self) -> int:
        pass

    @abstractmethod
    def generate_features(self, features_generators: list[str]) -> np.ndarray:
        pass

    def reset_features_and_targets(self):
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        self.features = self._features
        self.targets = self._targets


@dataclass
class MoleculeDatapointMixin:
    smi: str
    mol: Chem.Mol = field(init=False)


@dataclass
class MoleculeDatapoint(DatapointBase, MoleculeDatapointMixin):
    """A `MoleculeDatapoint` contains a single molecule and its associated features and targets.

    Parameters
    ----------
    smi : str
        the SMILES string of the molecule
    atom_features : Optional[np.ndarray], default=None
        a numpy array containing additional features that are concatenated to atom-level
        features *before* message passing
    bond_features : Optional[np.ndarray], default=None
        A numpy array containing additional features that are concatenated to bond-level
        features *before* message passing
    atom_descriptors : Optional[np.ndarray], default=None
        A numpy array containing additional features that are concatenated to atom-level
        features *after* message passing
    targets : Optional[np.ndarray], default=None
        the targets for the molecule with unknown targets indicated by `nan`s
    row : Optional[OrderedDict], default=None
        The raw CSV row containing the information for this molecule.
    data_weight : float, default=1
        Weighting of the datapoint for the loss function.
    gt_targets : Optional[np.ndarray], default=None
        Indicates whether the targets are an inequality regression target of the form `>x`
    lt_targets : Optional[np.ndarray], default=None
        Indicates whether the targets are an inequality regression target of the form `<x`
    features : Optional[np.ndarray], default=None
        A numpy array containing additional features (e.g., Morgan fingerprint).
    features_generators : Optional[List[str]], default=None
        A list of features generators to use
    phase_features : Optional[np.ndarray], default=None
        A one-hot vector indicating the phase of the data, as used in spectra data.
    keep_h : bool, default=False
        whether to retain the hydrogens present in input molecules or remove them from the prepared
        structure
    add_h : bool, default=False
        whether to add hydrogens to all input molecules when preparing the input structure

    Attributes
    ----------
    _all input parameters_
    mol : Chem.Mol
        the RDKit molecule of the input

    """

    atom_features: Optional[np.ndarray] = None
    bond_features: Optional[np.ndarray] = None
    atom_descriptors: Optional[np.ndarray] = None

    def __post_init__(self, features_generators: Optional[List[str]]):
        self.mol = make_mol(self.smi, self.explicit_h, self.add_h)

        replace_token = 0
        if self.atom_features is not None:
            self.atom_features[np.isnan(self.atom_features)] = replace_token
        if self.bond_features is not None:
            self.bond_features[np.isnan(self.bond_features)] = replace_token
        if self.atom_descriptors is not None:
            self.atom_descriptors[np.isnan(self.atom_descriptors)] = replace_token

        self._atom_features = self.atom_features
        self._bond_features = self.bond_features
        self._atom_descriptors = self.atom_descriptors

        super().__post_init__(features_generators)

    @property
    def number_of_molecules(self) -> int:
        return 1

    def generate_features(self, features_generators: list[str]) -> np.ndarray:
        features = []
        for fg in features_generators:
            fg = get_features_generator(fg)
            if self.mol is not None:
                if self.mol.GetNumHeavyAtoms() > 0:
                    features.append(fg(self.mol))
                else:
                    features.append(np.zeros(len(fg(Chem.MolFromSmiles("C")))))

        return np.hstack(features)

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        self.features = self._features
        self.targets = self._targets
        self.atom_descriptors = self._atom_descriptors
        self.atom_features = self._atom_features
        self.bond_features = self._bond_features


@dataclass
class ReactionDatapointMixin:
    smis: list[str]
    mols: list[Chem.Mol] = field(init=False)


class ReactionDatapoint(DatapointBase, ReactionDatapointMixin):
    """
    Parameters
    ----------
    smis : list[str]
        the SMILES strings of the reactants and products of the reaction
    mols : list[Chem.Mol]
        the rdkit molecules of the reactants and products of the reaction
    targets : np.ndarray
        the targets for the molecule with unknown targets indicated by `nan`s
    row : OrderedDict, default=None
        The raw CSV row containing the information for this molecule.
    data_weight : float, default=1
        Weighting of the datapoint for the loss function.
    gt_targets : Optional[np.ndarray], default=None
        Indicates whether the targets are an inequality regression target of the form `>x`
    lt_targets : Optional[np.ndarray], default=None
        Indicates whether the targets are an inequality regression target of the form `<x`
    features : Optional[np.ndarray], default=None
        A numpy array containing additional features (e.g., Morgan fingerprint).
    features_generators : Optional[List[str]], default=None
        A list of features generators to use
    phase_features : Optional[np.ndarray], default=None
        A one-hot vector indicating the phase of the data, as used in spectra data.
    atom_descriptors : Optional[np.ndarray], default=None
        A numpy array containing additional atom descriptors with which to featurize the molecule
    keep_h : bool, default=False
        whether to retain the hydrogens present in input molecules or remove them from the prepared
        structure
    add_h : bool, default=False
        whether to add hydrogens to all input molecules when preparing the input structure
    """

    def __post_init__(self, features_generators: Optional[List[str]]):
        self.mols = [make_mol(smi, self.explicit_h, self.add_h) for smi in self.smis]

        super().__post_init__(features_generators)

    @property
    def number_of_molecules(self) -> int:
        return len(self.smis)

    def generate_features(self, features_generators: list[str]) -> np.ndarray:
        features = []
        for fg in features_generators:
            fg = get_features_generator(fg)
            for mol in self.mols:
                if mol is not None:
                    if mol.GetNumHeavyAtoms() > 0:
                        features.append(fg(mol))
                    else:
                        features.append(np.zeros(len(fg(Chem.MolFromSmiles("C")))))

        return np.hstack(features)
