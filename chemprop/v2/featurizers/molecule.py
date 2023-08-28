from dataclasses import InitVar, dataclass

import numpy as np
from rdkit import Chem

from chemprop.v2.featurizers.protos import MoleculeMolGraphFeaturizerProto
from chemprop.v2.featurizers.mixins import MolGraphFeaturizerMixin
from chemprop.v2.featurizers.molgraph import MolGraph


@dataclass
class MoleculeMolGraphFeaturizer(MolGraphFeaturizerMixin, MoleculeMolGraphFeaturizerProto):
    """A :class:`MoleculeMolGraphFeaturizer` is the default implementation of a
    :class:`MoleculeMolGraphFeaturizerProto`

    Parameters
    ----------
    atom_featurizer : AtomFeaturizerProto, default=AtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizerProto, default=BondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    bond_messages : bool, default=True
        whether to prepare the `MolGraph`s for use with message passing on bonds
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

        self.atom_fdim += extra_atom_fdim
        self.bond_fdim += extra_bond_fdim
        if self.bond_messages:
            self.bond_fdim += self.atom_fdim

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

        X_v = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()])
        X_e = np.empty((2 * n_bonds, self.bond_fdim))
        a2b = [[] for _ in range(n_atoms)]
        b2a = np.empty(2 * n_bonds, int)
        b2revb = np.empty(2 * n_bonds, int)

        if atom_features_extra is not None:
            X_v = np.hstack((X_v, atom_features_extra))

        i = 0
        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue

                x_e = self.bond_featurizer(bond)
                if bond_features_extra is not None:
                    x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]))

                b12 = i
                b21 = b12 + 1

                if self.bond_messages:
                    X_e[b12] = np.concatenate((X_v[a1], x_e))
                    X_e[b21] = np.concatenate((X_v[a2], x_e))
                else:
                    X_e[b12] = x_e
                    X_e[b21] = x_e

                a2b[a2].append(b12)
                a2b[a1].append(b21)

                b2a[i : i + 2] = [a1, a2]
                b2revb[i : i + 2] = [b21, b12]

                i += 2

        return MolGraph(n_atoms, 2 * n_bonds, X_v, X_e, a2b, b2a, b2revb, None, None)
