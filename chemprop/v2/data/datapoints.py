from __future__ import annotations

from dataclasses import InitVar, dataclass

import numpy as np
from rdkit.Chem import AllChema as Chem

from chemprop.v2.utils import make_mol
from chemprop.featurizers import get_features_generator


@dataclass(slots=True)
class DatapointMixin:
    """A mixin class for both molecule- and reaction-type data"""

    y: np.ndarray | None = None
    """the targets for the molecule with unknown targets indicated by `nan`s"""
    weight: float = 1
    """the weight of this datapoint for the loss calculation."""
    gt_mask: np.ndarray | None = None
    """Indicates whether the targets are an inequality regression target of the form `<x`"""
    lt_mask: np.ndarray | None = None
    """Indicates whether the targets are an inequality regression target of the form `>x`"""
    x_f: np.ndarray | None = None
    """A vector of length ``d_f`` containing additional features (e.g., Morgan fingerprint) that
    will be concatenated to the global representation *after* aggregation"""
    features_generators: InitVar[list[str] | None] = None
    """A list of features generators to use"""
    x_phase: list[float] = None
    """A one-hot vector indicating the phase of the data, as used in spectra data."""

    def __post_init__(self, fgs: list[str] | None):
        if self.x_f is not None and fgs is not None:
            raise ValueError("Cannot provide both loaded features and features generators!")

        if fgs is not None:
            self.x_f = self.generate_features(fgs)

        NAN_TOKEN = 0
        if self.x_f is not None:
            self.x_f[np.isnan(self.x_f)] = NAN_TOKEN

    @property
    def t(self) -> int | None:
        return len(self.y) if self.y is not None else None


@dataclass
class MoleculeDatapointMixin:
    mol: Chem.Mol
    """the molecule associated with this datapoint"""

    @classmethod
    def from_smi(
        cls, smi: str, keep_h: bool = False, add_h: bool = False, *args, **kwargs
    ) -> MoleculeDatapointMixin:
        mol = make_mol(smi, keep_h, add_h)

        return cls(mol, *args, **kwargs)


@dataclass
class MoleculeDatapoint(DatapointMixin, MoleculeDatapointMixin):
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    V_f: np.ndarray | None = None
    """a numpy array of shape `V x d_vf`, where `V` is the number of atoms in the molecule, and
    `d_vf` is the number of additional features that will be concatenated to atom-level features
    _before_ message passing"""
    E_f: np.ndarray | None = None
    """A numpy array of shape `E x d_ef`, where `E` is the number of bonds in the molecule, and
    `d_ef` is the number of additional features  containing additional features that will be
    concatenated to bond-level features _before_ message passing"""
    V_d: np.ndarray | None = None
    """A numpy array of shape `V x d_vd`, where `V` is the number of atoms in the molecule, and
    `d_vd` is the number of additional features that will be concatenated to atom-level features
    _after_ message passing"""

    def __post_init__(self, features_generators: list[str | None]):
        NAN_TOKEN = 0

        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = NAN_TOKEN
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = NAN_TOKEN
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = NAN_TOKEN

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


@dataclass
class ReactionDatapointMixin:
    rct: Chem.Mol
    """the reactant associated with this datapoint"""
    pdt: Chem.Mol
    """the product associated with this datapoint"""

    @classmethod
    def from_smi(
        cls,
        rxn_or_smis: str | tuple[str, str],
        keep_h: bool = False,
        add_h: bool = False,
        *args,
        **kwargs,
    ) -> ReactionDatapointMixin:
        match rxn_or_smis:
            case str():
                # rxn = Chem.ReactionFromSmarts(rxn_or_smis)
                rct_smi, pdt_smi = rxn_or_smis.split(">>")
            case tuple():
                rct_smi, pdt_smi = rxn_or_smis
            case _:
                raise TypeError(
                    "Must provide either a reaction SMARTS string or a tuple of reactant and product SMILES strings!"
                )

        rct = make_mol(rct_smi, keep_h, add_h)
        pdt = make_mol(pdt_smi, keep_h, add_h)

        return cls(rct, pdt, *args, **kwargs)


@dataclass
class ReactionDatapoint(DatapointMixin, ReactionDatapointMixin):
    """A :class:`ReactionDatapoint` contains a single reaction and its associated features and targets."""
    def __post_init__(self, features_generators: list[str] | None):
        self.rct = make_mol(self.rct_smi, self.explicit_h, self.add_h)
        self.pdt = make_mol(self.pdt_smi, self.explicit_h, self.add_h)

        super().__post_init__(features_generators)

    def __len__(self) -> int:
        return 2

    def generate_features(self, features_generators: list[str]) -> np.ndarray:
        features = []
        for fg in features_generators:
            fg = get_features_generator(fg)
            for mol in [self.rct, self.pdt]:
                if mol is not None:
                    if mol.GetNumHeavyAtoms() > 0:
                        features.append(fg(mol))
                    else:
                        features.append(np.zeros(len(fg(Chem.MolFromSmiles("C")))))

        return np.hstack(features)


@dataclass
class MulticomponentDatapointMixin:
    mols: list[Chem.Mol]
    """the molecules associated with this datapoint"""

    @classmethod
    def from_smis(
        cls,
        smis: list[str],
        keep_h: bool = False,
        add_h: bool = False,
        *args,
        **kwargs,
    ) -> MulticomponentDatapointMixin:
        mols = [make_mol(smi, keep_h, add_h) for smi in smis]

        return cls(mols, *args, **kwargs)


@dataclass
class MulticomponentDatapoint(DatapointMixin, MulticomponentDatapointMixin):
    """A :class:`MulticomponentDatapoint` contains a list of molecules and their associated features and targets."""
    def __len__(self) -> int:
        return len(self.mols)

    def generate_features(self, features_generators: list[str]) -> np.ndarray:
        Xs = []
        for fg in features_generators:
            fg = get_features_generator(fg)
            for mol in self.mols:
                if mol is not None:
                    if mol.GetNumHeavyAtoms() > 0:
                        Xs.append(fg(mol))
                    else:
                        Xs.append(np.zeros(len(fg(Chem.MolFromSmiles("C")))))

        return np.hstack(Xs)
