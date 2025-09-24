from __future__ import annotations

from dataclasses import dataclass, field

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
    x_phase: list[float] = None
    """A one-hot vector indicating the phase of the data, as used in spectra data."""
    name: str | None = None
    """A string identifier for the datapoint."""

    def __post_init__(self):
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
        cls,
        smi: str,
        *args,
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = False,
        **kwargs,
    ) -> _MoleculeDatapointMixin:
        mol = make_mol(smi, keep_h, add_h, ignore_stereo, reorder_atoms)

        kwargs["name"] = smi if "name" not in kwargs else kwargs["name"]

        return cls(mol, *args, **kwargs)


@dataclass
class _LazyMoleculeDatapointMixin:
    smiles: str
    """the SMILES string associated with this datapoint"""
    _keep_h: bool = False
    _add_h: bool = False
    _ignore_stereo: bool = False
    _reorder_atoms: bool = False
    _mol_cache: Chem.Mol = field(default=None, repr=False, compare=False)

    @property
    def mol(self) -> Chem.Mol:
        """Lazily compute the molecule only when accessed"""
        if self._mol_cache is None:
            self._mol_cache = make_mol(
                self.smiles, self._keep_h, self._add_h, self._ignore_stereo, self._reorder_atoms
            )
        return self._mol_cache


@dataclass
class MoleculeDatapoint(_DatapointMixin, _MoleculeDatapointMixin):
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    V_f: np.ndarray | None = None
    """A numpy array of shape ``V x d_vf``, where ``V`` is the number of atoms in the molecule, and
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

    def __post_init__(self):
        NAN_TOKEN = 0
        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = NAN_TOKEN
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = NAN_TOKEN
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = NAN_TOKEN

        super().__post_init__()

    def __len__(self) -> int:
        return 1


@dataclass
class LazyMoleculeDatapoint(_DatapointMixin, _LazyMoleculeDatapointMixin):
    """A :class:`LazyMoleculeDatapoint` contains a single SMILES string, and all attributes need to
    form a `rdkit.Chem.Mol` object. The molecule is computed lazily when the attribute `mol` is accessed.
    """

    V_f: np.ndarray | None = None
    """A numpy array of shape ``V x d_vf``, where ``V`` is the number of atoms in the molecule, and
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

    def __post_init__(self):
        NAN_TOKEN = 0
        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = NAN_TOKEN
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = NAN_TOKEN
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = NAN_TOKEN

        super().__post_init__()

    def __len__(self) -> int:
        return 1


@dataclass
class MolAtomBondDatapoint(MoleculeDatapoint):
    E_d: np.ndarray | None = None
    """A numpy array of shape ``E x d_ed``, where ``E`` is the number of bonds in the molecule, and
    ``d_ed`` is the number of additional descriptors that will be concatenated to edge-level
    descriptors *after* message passing"""
    atom_y: np.ndarray | None = None
    """A numpy array of shape ``V x v_t``, where ``V`` is the number of atoms in the molecule, and
    ``v_t`` is the number of atom targets. The order of atoms in the array should match the order of
    atoms in the mol. Unknown targets are indicated by `nan`s."""
    atom_gt_mask: np.ndarray | None = None
    """Indicates whether the atom targets are an inequality regression target of the form `<x`"""
    atom_lt_mask: np.ndarray | None = None
    """Indicates whether the atom targets are an inequality regression target of the form `>x`"""
    bond_y: np.ndarray | None = None
    """A numpy array of shape ``E x e_t``, where ``V`` is the number of bonds in the molecule, and
    ``e_t`` is the number of bond targets. The order of bonds in the array should match the order of
    bonds in the mol. Unknown targets are indicated by `nan`s."""
    bond_gt_mask: np.ndarray | None = None
    """Indicates whether the bond targets are an inequality regression target of the form `<x`"""
    bond_lt_mask: np.ndarray | None = None
    """Indicates whether the bond targets are an inequality regression target of the form `>x`"""
    atom_constraint: np.ndarray | None = None
    """A numpy array of shape ``1 x v_t`` containing the values that the atom property predictions
    should be constrained to sum to, with np.nan indicating no constraint for that property"""
    bond_constraint: np.ndarray | None = None
    """A numpy array of shape ``1 x e_t`` containing the values that the bond property predictions
    should be constrained to sum to, with np.nan indicating no constraint for that property"""

    def __post_init__(self):
        super().__post_init__()
        NAN_TOKEN = 0
        if self.E_d is not None:
            self.E_d[np.isnan(self.E_d)] = NAN_TOKEN

    @classmethod
    def from_smi(
        cls,
        smi: str,
        *args,
        keep_h: bool = False,
        add_h: bool = False,
        ignore_stereo: bool = False,
        reorder_atoms: bool = True,
        **kwargs,
    ) -> MolAtomBondDatapoint:
        mol = make_mol(smi, keep_h, add_h, ignore_stereo, reorder_atoms=reorder_atoms)

        kwargs["name"] = smi if "name" not in kwargs else kwargs["name"]

        return cls(mol, *args, **kwargs)


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
        ignore_stereo: bool = False,
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
                    "Must provide either a reaction SMARTS string or a tuple of reactant and"
                    " a product SMILES strings!"
                )

        rct = make_mol(rct_smi, keep_h, add_h, ignore_stereo)
        pdt = make_mol(pdt_smi, keep_h, add_h, ignore_stereo)

        kwargs["name"] = name if "name" not in kwargs else kwargs["name"]

        return cls(rct, pdt, *args, **kwargs)


@dataclass
class ReactionDatapoint(_DatapointMixin, _ReactionDatapointMixin):
    """A :class:`ReactionDatapoint` contains a single reaction and its associated features and targets."""

    def __post_init__(self):
        if self.rct is None:
            raise ValueError("Reactant cannot be `None`!")
        if self.pdt is None:
            raise ValueError("Product cannot be `None`!")

        return super().__post_init__()

    def __len__(self) -> int:
        return 2
