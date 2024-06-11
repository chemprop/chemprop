from __future__ import annotations

from dataclasses import InitVar, dataclass

import numpy as np
from rdkit.Chem import AllChem as Chem

from chemprop.featurizers import Featurizer
from chemprop.utils import make_mol

MoleculeFeaturizer = Featurizer[Chem.Mol, np.ndarray]


@dataclass(slots=True)
class _DatapointMixin:
    """A mixin class for both molecule- and reaction- and multicomponent-type data"""

    y: np.ndarray | None = None
    """the targets for the molecule with unknown targets indicated by `nan`s"""
    weight: float = 1.0
    """the weight of this datapoint for the loss calculation."""
    gt_mask: np.ndarray | None = None
    """Indicates whether the targets are an inequality regression target of the form `<x`"""
    lt_mask: np.ndarray | None = None
    """Indicates whether the targets are an inequality regression target of the form `>x`"""
    x_d: np.ndarray | None = None
    """A vector of length ``d_f`` containing additional features (e.g., Morgan fingerprint) that
    will be concatenated to the global representation *after* aggregation"""
    mfs: InitVar[list[MoleculeFeaturizer] | None] = None
    """A list of molecule featurizers to use"""
    x_phase: list[float] = None
    """A one-hot vector indicating the phase of the data, as used in spectra data."""
    name: str | None = None
    """A string identifier for the datapoint."""

    def __post_init__(self, mfs: list[MoleculeFeaturizer] | None):
        if self.x_d is not None and mfs is not None:
            raise ValueError("Cannot provide both loaded features and molecular featurizers!")

        if mfs is not None:
            self.x_d = self.calc_features(mfs)

        NAN_TOKEN = 0
        if self.x_d is not None:
            self.x_d[np.isnan(self.x_d)] = NAN_TOKEN

    @property
    def t(self) -> int | None:
        return len(self.y) if self.y is not None else None


@dataclass
class _MoleculeDatapointMixin:
    mol: Chem.Mol
    """the molecule associated with this datapoint"""

    @classmethod
    def from_smi(
        cls, smi: str, *args, keep_h: bool = False, add_h: bool = False, **kwargs
    ) -> _MoleculeDatapointMixin:
        mol = make_mol(smi, keep_h, add_h)

        kwargs["name"] = smi if "name" not in kwargs else kwargs["name"]

        return cls(mol, *args, **kwargs)


@dataclass
class MoleculeDatapoint(_DatapointMixin, _MoleculeDatapointMixin):
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    V_f: np.ndarray | None = None
    """a numpy array of shape ``V x d_vf``, where ``V`` is the number of atoms in the molecule, and
    ``d_vf`` is the number of additional features that will be concatenated to atom-level features
    *before* message passing"""
    E_f: np.ndarray | None = None
    """A numpy array of shape ``E x d_ef``, where ``E`` is the number of bonds in the molecule, and
    ``d_ef`` is the number of additional features  containing additional features that will be
    concatenated to bond-level features *before* message passing"""
    V_d: np.ndarray | None = None
    """A numpy array of shape ``V x d_vd``, where ``V`` is the number of atoms in the molecule, and
    ``d_vd`` is the number of additional descriptors that will be concatenated to atom-level
    descriptors *after* message passing"""

    def __post_init__(self, mfs: list[MoleculeFeaturizer] | None):
        if self.mol is None:
            raise ValueError("Input molecule was `None`!")

        NAN_TOKEN = 0

        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = NAN_TOKEN
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = NAN_TOKEN
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = NAN_TOKEN

        super().__post_init__(mfs)

    def __len__(self) -> int:
        return 1

    def calc_features(self, mfs: list[MoleculeFeaturizer]) -> np.ndarray:
        if self.mol.GetNumHeavyAtoms() == 0:
            return np.zeros(sum(len(mf) for mf in mfs))

        return np.hstack([mf(self.mol) for mf in mfs])


@dataclass
class _ReactionDatapointMixin:
    rct: Chem.Mol
    """the reactant associated with this datapoint"""
    pdt: Chem.Mol
    """the product associated with this datapoint"""

    @classmethod
    def from_smi(
        cls,
        rxn_or_smis: str | tuple[str, str],
        *args,
        keep_h: bool = False,
        add_h: bool = False,
        **kwargs,
    ) -> _ReactionDatapointMixin:
        match rxn_or_smis:
            case str():
                rct_smi, agt_smi, pdt_smi = rxn_or_smis.split(">")
                rct_smi = f"{rct_smi}.{agt_smi}" if agt_smi else rct_smi
                name = rxn_or_smis
            case tuple():
                rct_smi, pdt_smi = rxn_or_smis
                name = ">>".join(rxn_or_smis)
            case _:
                raise TypeError(
                    "Must provide either a reaction SMARTS string or a tuple of reactant and product SMILES strings!"
                )

        rct = make_mol(rct_smi, keep_h, add_h)
        pdt = make_mol(pdt_smi, keep_h, add_h)

        kwargs["name"] = name if "name" not in kwargs else kwargs["name"]

        return cls(rct, pdt, *args, **kwargs)


@dataclass
class ReactionDatapoint(_DatapointMixin, _ReactionDatapointMixin):
    """A :class:`ReactionDatapoint` contains a single reaction and its associated features and targets."""

    def __post_init__(self, mfs: list[MoleculeFeaturizer] | None):
        if self.rct is None:
            raise ValueError("Reactant cannot be `None`!")
        if self.pdt is None:
            raise ValueError("Product cannot be `None`!")

        return super().__post_init__(mfs)

    def __len__(self) -> int:
        return 2

    def calc_features(self, mfs: list[MoleculeFeaturizer]) -> np.ndarray:
        x_ds = [
            mf(mol) if mol.GetNumHeavyAtoms() > 0 else np.zeros(len(mf))
            for mf in mfs
            for mol in [self.rct, self.pdt]
        ]

        return np.hstack(x_ds)
