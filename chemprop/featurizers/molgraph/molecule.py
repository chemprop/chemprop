from dataclasses import InitVar, dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import GraphFeaturizer
from chemprop.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin
from chemprop.utils.utils import is_cuikmolmaker_available

if is_cuikmolmaker_available():
    import cuik_molmaker


@dataclass
class SimpleMoleculeMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """A :class:`SimpleMoleculeMolGraphFeaturizer` is the default implementation of a
    :class:`MoleculeMolGraphFeaturizer`

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
    """

    extra_atom_fdim: InitVar[int] = 0
    extra_bond_fdim: InitVar[int] = 0

    def __post_init__(self, extra_atom_fdim: int = 0, extra_bond_fdim: int = 0):
        super().__post_init__()

        self.extra_atom_fdim = extra_atom_fdim
        self.extra_bond_fdim = extra_bond_fdim
        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)
        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        i = 0
        for bond in mol.GetBonds():
            x_e = self.bond_featurizer(bond)
            if bond_features_extra is not None:
                x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single)

            E[i : i + 2] = x_e

            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])

            i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return MolGraph(V, E, edge_index, rev_edge_index)


@dataclass
class CuikmolmakerMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """A :class:`CuikmolmakerMolGraphFeaturizer` is the default implementation of a
    :class:`_MolGraphFeaturizerMixin`. This class featurizes a list of molecules at once instead of one molecule at a time for efficiency.

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
    atom_featurizer_mode: str, default="V2"
        The mode of the atom featurizer (V1, V2, ORGANIC) to use.
    """

    atom_featurizer_mode: str = "V2"
    add_h: bool = False

    def __post_init__(self, atom_featurizer_mode: str = "V2", add_h: bool = False):
        super().__post_init__()
        if not is_cuikmolmaker_available():
            raise ImportError(
                "CuikmolmakerMolGraphFeaturizer requires cuik-molmaker package to be installed. "
                f"Please install it using `python {Path(__file__).parents[1] / Path('scripts/check_and_install_cuik_molmaker.py')}`."
            )
        bond_props = ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"]

        if self.atom_featurizer_mode == "V1":
            atom_props_onehot = [
                "atomic-number",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization",
            ]
        elif self.atom_featurizer_mode == "V2":
            atom_props_onehot = [
                "atomic-number-common",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization-expanded",
            ]
        elif self.atom_featurizer_mode == "ORGANIC":
            atom_props_onehot = [
                "atomic-number-organic",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization-organic",
            ]
        elif self.atom_featurizer_mode == "RIGR":
            atom_props_onehot = ["atomic-number-common", "total-degree", "num-hydrogens"]
            bond_props = ["is-null", "in-ring"]
        else:
            raise ValueError(f"Invalid atom featurizer mode: {atom_featurizer_mode}")

        self.atom_property_list_onehot = cuik_molmaker.atom_onehot_feature_names_to_tensor(
            atom_props_onehot
        )

        atom_props_float = ["aromatic", "mass"]
        self.atom_property_list_float = cuik_molmaker.atom_float_feature_names_to_tensor(
            atom_props_float
        )

        self.bond_property_list = cuik_molmaker.bond_feature_names_to_tensor(bond_props)

    def __call__(
        self,
        smiles_list: list[str],
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ):
        offset_carbon, duplicate_edges, add_self_loop = False, True, False

        batch_feats = cuik_molmaker.batch_mol_featurizer(
            smiles_list,
            self.atom_property_list_onehot,
            self.atom_property_list_float,
            self.bond_property_list,
            self.add_h,
            offset_carbon,
            duplicate_edges,
            add_self_loop,
        )
        return batch_feats
