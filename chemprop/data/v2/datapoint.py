from collections import OrderedDict
from dataclasses import InitVar, dataclass, field
from typing import List, Optional

import numpy as np
from rdkit import Chem

from chemprop.rdkit import make_mol
from chemprop.featurizers import get_features_generator


@dataclass
class MoleculeDatapoint:
    """A `MoleculeDatapoint` contains a single molecule and its associated features and targets.
    
    Parameters
    ----------
    smiles : str
        the SMILES string of the molecule
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
        A numpy array containing additional atom descriptors to featurize the molecule
    bond_features : Optional[np.ndarray], default=None
        A numpy array containing additional bond features to featurize the molecule
    keep_h : bool, default=False
        whether to retain the hydrogens present in input molecules or remove them from the prepared 
        structure
    add_h : bool, default=False
        whether to add hydrogens to all input molecules when preparing the input structure
    """

    smiles: str
    targets: np.ndarray
    mol: Chem.Mol = field(init=False)
    row: OrderedDict = None
    data_weight: float = 1
    gt_targets: List[bool] = None
    lt_targets: List[bool] = None
    features: Optional[np.ndarray] = None
    features_generators: InitVar[Optional[List[str]]] = None
    phase_features: List[float] = None
    atom_features: np.ndarray = None
    atom_descriptors: np.ndarray = None
    bond_features: np.ndarray = None
    explicit_h: bool = False
    add_h: bool = False

    def __post_init__(self, features_generators: Optional[List[str]]):
        if self.features is not None and features_generators is not None:
            raise ValueError("Cannot provide both loaded features and features generators!")

        self.mol = make_mol(self.smiles, self.explicit_h, self.add_h)

        if features_generators is not None:
            self.features = []
            for fg in features_generators:
                fg = get_features_generator(fg)
                if self.mol is not None:
                    if self.mol.GetNumHeavyAtoms() > 0:
                        self.features.extend(fg(self.mol))
                    else:
                        self.features.extend(np.zeros(len(fg(Chem.MolFromSmiles("C")))))
            self.features = np.array(self.features)

        replace_token = 0
        if self.features is not None:
            self.features[np.isnan(self.features)] = replace_token

        if self.atom_descriptors is not None:
            self.atom_descriptors[np.isnan(self.atom_descriptors)] = replace_token

        if self.atom_features is not None:
            self.atom_features[np.isnan(self.atom_features)] = replace_token

        if self.bond_features is not None:
            self.bond_features[np.isnan(self.bond_features)] = replace_token

        self.raw_features = self.features
        self.raw_targets = self.targets
        self.raw_atom_descriptors = self.atom_descriptors 
        self.raw_atom_features = self.atom_features
        self.raw_bond_features = self.bond_features

    @property
    def number_of_molecules(self) -> int:
        return 1

    @property
    def num_tasks(self) -> int:
        return len(self.targets)

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        self.features = self.raw_features
        self.targets = self.raw_targets
        self.atom_descriptors = self.raw_atom_descriptors
        self.atom_features = self.raw_atom_features
        self.bond_features = self.raw_bond_features
