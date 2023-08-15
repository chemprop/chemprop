from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import InitVar, dataclass, field

import numpy as np
from rdkit import Chem

from chemprop.v2.utils import make_mol
from chemprop.featurizers import get_features_generator


@dataclass
class DatapointBase(ABC):
    """A `DatapointBase` is the base datapoint for both molecule- and reaction-type data"""

    y: np.ndarray | None = None
    row: OrderedDict = None
    weight: float = 1
    gt_mask: np.ndarray | None = None
    lt_mask: np.ndarray | None = None
    x_v: np.ndarray | None = None
    features_generators: InitVar[list[str] | None] = None
    x_phase: list[float] = None
    explicit_h: bool = False
    add_h: bool = False

    def __post_init__(self, fgs: list[str] | None):
        if self.x_v is not None and fgs is not None:
            raise ValueError("Cannot provide both loaded features and features generators!")

        if fgs is not None:
            self.x_v = self.generate_features(fgs)

        if self.x_v is not None:
            NAN_TOKEN = 0
            self.x_v[np.isnan(self.x_v)] = NAN_TOKEN

        self._x_v = self.x_v
        self._y = self.y

    @abstractmethod
    def __len__(self) -> int:
        """the number of molecules in this datapoint"""

    @property
    def t(self) -> int | None:
        return len(self.y) if self.y is not None else None

    @abstractmethod
    def generate_features(self, features_generators: list[str]) -> np.ndarray:
        pass

    def reset(self):
        """Reset the molecule features and targets of each datapoint to its 
        initial, unnormalized values."""
        self.x_v = self._x_v
        self.y = self._y


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
    y : np.ndarray | None, default=None
        the targets for the molecule with unknown targets indicated by `nan`s
    row : OrderedDict | None, default=None
        The raw CSV row containing the information for this molecule.
    weight : float, default=1
        the weight of this datapoint for the loss calculation.
    lt_mask : np.ndarray | None, default=None
        Indicates whether the targets are an inequality regression target of the form `<x`
    gt_mask : np.ndarray | None, default=None
        Indicates whether the targets are an inequality regression target of the form `>x`
    x_v : np.ndarray | None, default=None
        A vector of length `d_v` containing additional features (e.g., Morgan fingerprint) that will
        be concatenated to the global representation _after_ aggregation
    features_generators : list[str | None], default=None
        A list of features generators to use
    x_phase : np.ndarray | None, default=None
        A one-hot vector indicating the phase of the data, as used in spectra data.
    keep_h : bool, default=False
        whether to retain the hydrogens present in input molecules or remove them from the prepared
        structure
    add_h : bool, default=False
        whether to add hydrogens to all input molecules when preparing the input structure
    V_f : np.ndarray | None, default=None
        a numpy array of shape `V x d_vf`, where `V` is the number of atoms in the molecule, and
        `d_vf` is the number of additional features that will be concatenated to atom-level features
        _before_ message passing
    E_f : np.ndarray | None, default=None
        A numpy array of shape `E x d_ef`, where `E` is the number of bonds in the molecule, and
        `d_ef` is the number of additional features  containing additional features that will be
        concatenated to bond-level features _before_ message passing
    V_d : np.ndarray | None, default=None
        A numpy array of shape `V x d_vd`, where `V` is the number of atoms in the molecule, and
        `d_vd` is the number of additional features that will be concatenated to atom-level features
        _after_ message passing

    Attributes
    ----------
    _all input parameters_
    mol : Chem.Mol
        the RDKit molecule of the input

    """

    V_f: np.ndarray | None = None
    E_f: np.ndarray | None = None
    V_d: np.ndarray | None = None

    def __post_init__(self, features_generators: list[str | None]):
        self.mol = make_mol(self.smi, self.explicit_h, self.add_h)

        replace_token = 0
        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = replace_token
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = replace_token
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = replace_token

        self._V_f = self.V_f
        self._E_f = self.E_f
        self._V_d = self.V_d

        super().__post_init__(features_generators)

    def __len__(self) -> int:
        return 1

    def generate_features(self, fgs: list[str]) -> np.ndarray:
        features = []
        for fg in fgs:
            fg = get_features_generator(fg)
            if self.mol is not None:
                if self.mol.GetNumHeavyAtoms() > 0:
                    features.append(fg(self.mol))
                else:
                    features.append(np.zeros(len(fg(Chem.MolFromSmiles("C")))))

        return np.hstack(features)

    def reset(self) -> None:
        """Reset the {atom, bond, molecule} features and targets of each datapoint to its 
        initial, unnormalized values."""
        super().reset()
        self.V_d = self._V_d
        self.E_f = self._E_f
        self.V_f = self._V_f


@dataclass
class MulticomponentDatapointMixin:
    smis: list[str]
    mols: list[Chem.Mol] = field(init=False)


@dataclass
class MulticomponentDatapoint(DatapointBase, MulticomponentDatapointMixin):
    """
    Parameters
    ----------
    smis : list[str]
        the SMILES strings of the reactants and products of the reaction
    y : np.ndarray
        the targets for the molecule with unknown targets indicated by `nan`s
    row : OrderedDict, default=None
        The raw CSV row containing the information for this molecule.
    weight : float, default=1
        the weight of this datapoint for the loss calculation.
    gt_mask : np.ndarray | None, default=None
        Indicates whether the targets are an inequality regression target of the form `>x`
    lt_mask : np.ndarray | None, default=None
        Indicates whether the targets are an inequality regression target of the form `<x`
    x_v : np.ndarray | None, default=None
        A vector of length `d_v` containing additional features (e.g., Morgan fingerprint) that will
        be concatenated to the global representation _after_ aggregation
    features_generators : list[str] | None, default=None
        A list of features generators to use
    x_phase : np.ndarray | None, default=None
        A one-hot vector indicating the phase of the data, as used in spectra data.
    keep_h : bool, default=False
        whether to retain the hydrogens present in input molecules or remove them from the prepared
        structure
    add_h : bool, default=False
        whether to add hydrogens to all input molecules when preparing the input structure

    Attributes
    ----------
    _all input parameters_
    mols : list[Chem.Mol]
        the RDKit molecules of the reactants and products of the reaction
    """

    def __post_init__(self, features_generators: list[str] | None):
        self.mols = [make_mol(smi, self.explicit_h, self.add_h) for smi in self.smis]

        super().__post_init__(features_generators)

    def __len__(self) -> int:
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
