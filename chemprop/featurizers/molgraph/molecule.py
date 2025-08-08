from dataclasses import InitVar, dataclass
import itertools

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import GraphFeaturizer
from chemprop.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin


@dataclass
class SimpleMoleculeMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    r"""A :class:`SimpleMoleculeMolGraphFeaturizer` is the default implementation of a
    :class:`MoleculeMolGraphFeaturizer`

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    backward_bond_featurizer : BondFeaturizer | None, default=None
        the featurizer with which to compute feature representations for backward bonds in a
        molecule. If this is ``None``, the ``bond_featurizer`` will be used for both forward and
        backward bonds. A forward bond is defined as starting at ``bond.GetBeginAtom()`` and ending
        at ``bond.GetEndAtom()``, while a reversed bond starts at ``bond.GetEndAtom()`` and ends at
        ``bond.GetBeginAtom()``.
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond

    Example
    -------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C=CO")
    >>> featurizer = SimpleMoleculeMolGraphFeaturizer()
    >>> print(featurizer.to_string(mol))
    0: 00000100000000000000000000000000000000 0001000 000010 10000 001000 00100000 0 0.120
    1: 00000100000000000000000000000000000000 0001000 000010 10000 010000 00100000 0 0.120
    2: 00000001000000000000000000000000000000 0010000 000010 10000 010000 00100000 0 0.160
    0→1: 0 0100 1 0 1000000
    0←1: 0 0100 1 0 1000000
    1→2: 0 1000 1 0 1000000
    1←2: 0 1000 1 0 1000000

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
        backward_bond_features_extra: np.ndarray | None = None,
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
        if backward_bond_features_extra is None:
            backward_bond_features_extra = bond_features_extra
        elif bond_features_extra is None:
            raise ValueError(
                "`backward_bond_features_extra` requires `bond_features_extra` to be provided!"
            )
        elif backward_bond_features_extra.shape != bond_features_extra.shape:
            raise ValueError(
                "`backward_bond_features_extra` and `bond_features_extra` must have the same shape!"
                f"got: {backward_bond_features_extra.shape} and {bond_features_extra.shape}"
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
            idx = bond.GetIdx()
            x_e_forward = self.bond_featurizer(bond)
            E[i] = (
                x_e_forward
                if bond_features_extra is None
                else np.concatenate((x_e_forward, bond_features_extra[idx]), dtype=np.single)
            )

            x_e_backward = (
                self.backward_bond_featurizer(bond)
                if self.backward_bond_featurizer is not None
                else x_e_forward
            )
            E[i + 1] = (
                x_e_backward
                if backward_bond_features_extra is None
                else np.concatenate(
                    (x_e_backward, backward_bond_features_extra[idx]), dtype=np.single
                )
            )

            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])

            i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return MolGraph(V, E, edge_index, rev_edge_index)

    def to_string(self, mol: Chem.Mol, decimals: int = 3) -> str:
        """
        Returns a string representation of the molecule featurization.

        Parameters
        ----------
        mol : Chem.Mol
            The RDKit molecule to featurize.
        decimals : int, optional
            The number of decimal places to round float-valued features to. Defaults to 3.

        Returns
        -------
        str
            A string representation of the molecule featurization, with each atom
            and bond represented on a separate line. The atom lines are of the form
            "i: <atom_features>" and the bond lines are of the form
            "i→j: <bond_features>" (forward directional bond) and "i←j: <bond_features>" (reverse bond).
        """
        n = mol.GetNumAtoms()
        digits = len(str(n))
        lines = []
        for i in range(n):
            string = self.atom_featurizer.to_string(mol.GetAtomWithIdx(i), decimals)
            lines.append(f"{i:{digits}}: {string}")
        for i, j in itertools.combinations(range(n), 2):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                string = self.bond_featurizer.to_string(bond, decimals)
                lines.append(f"{i:{digits}}\u2192{j:{digits}}: {string}")
                if self.backward_bond_featurizer is not None:
                    string = self.backward_bond_featurizer.to_string(bond, decimals)
                lines.append(f"{i:{digits}}\u2190{j:{digits}}: {string}")
        return "\n".join(lines)
