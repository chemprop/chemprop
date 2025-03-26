from dataclasses import InitVar, dataclass

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
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond

    Example
    -------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("O\C=C\O")
    >>> featurizer = SimpleMoleculeMolGraphFeaturizer()
    >>> print(featurizer.prettify(featurizer(mol)))
    0: 00000001000000000000000000000000000000 0010000 000010 10000 010000 00100000 0 0.160
    1: 00000100000000000000000000000000000000 0001000 000010 10000 010000 00100000 0 0.120
    2: 00000100000000000000000000000000000000 0001000 000010 10000 010000 00100000 0 0.120
    3: 00000001000000000000000000000000000000 0010000 000010 10000 010000 00100000 0 0.160
    0→1: 0 1000 1 0 1000000
    1→0: 0 1000 1 0 1000000
    1→2: 0 0100 1 0 0001000
    2→1: 0 0100 1 0 0001000
    2→3: 0 1000 1 0 1000000
    3→2: 0 1000 1 0 1000000

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

    def prettify(self, molgraph: MolGraph) -> str:
        """Convert the featurized molecule into a human-readable string."""
        vertices = "\n".join(
            f"{i}: {self.atom_featurizer.prettify(v)}" for i, v in enumerate(molgraph.V)
        )
        edges = "\n".join(
            f"{i}\u2192{j}: {self.bond_featurizer.prettify(e)}"
            for (i, j), e in zip(molgraph.edge_index.T, molgraph.E)
        )
        return "\n".join((vertices, edges))
