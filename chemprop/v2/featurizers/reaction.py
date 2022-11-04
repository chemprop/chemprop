from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Bond, Mol
from chemprop.v2.featurizers.utils import ReactionMode

from chemprop.v2.featurizers.atom import AtomFeaturizer
from chemprop.v2.featurizers.bond import BondFeaturizerBase
from chemprop.v2.featurizers.mixins import MolGraphFeaturizerMixin
from chemprop.v2.featurizers.molgraph import MolGraph
from chemprop.v2.featurizers.base import ReactionFeaturizerBase


class ReactionFeaturizer(MolGraphFeaturizerMixin, ReactionFeaturizerBase):
    """Featurize reactions using the condensed reaction graph method utilized in [1]_

    NOTE: This class *does not* accept a `BaseAtomFeaturizer` instance. This is because it requries
    the `featurize_num_only` method, which is only implemented in the concrete `AtomFeaturizer`
    class

    Attributes
    ----------
    atom_featurizer : AtomFeaturizer
    bond_featurizer : BondFeaturizerBase
    atom_fdim : int
        the dimension of atom feature represenatations in this featurizer
    bond_fdim : int
        the dimension of bond feature represenatations in this featurizer
    atom_messages : bool

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=AtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizerBase, default=BondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    bond_messages : bool, default=True
        whether to prepare the `MolGraph`s for use with bond-based message-passing
    mode : Union[str, ReactionMode], default=ReactionMode.REAC_DIFF
        the mode by which to featurize the reaction as either the string code or enum value

    References
    ----------
    .. [1] Heid, E.; Green, W.H. "Machine Learning of Reaction Properties via Learned
    Representations of the Condensed Graph of Reaction." J. Chem. Inf. Model. 2022, 62, 2101-2110.
    https://doi.org/10.1021/acs.jcim.1c00975
    """

    def __init__(
        self,
        atom_featurizer: Optional[AtomFeaturizer] = None,
        bond_featurizer: Optional[BondFeaturizerBase] = None,
        bond_messages: bool = True,
        mode: Union[str, ReactionMode] = ReactionMode.REAC_DIFF,
    ):
        super().__init__(atom_featurizer, bond_featurizer, bond_messages)

        self.mode = mode
        self.atom_fdim += len(self.atom_featurizer) - self.atom_featurizer.max_atomic_num - 1
        self.bond_fdim *= 2
        if self.bond_messages:
            self.bond_fdim += self.atom_fdim

    @property
    def mode(self) -> ReactionMode:
        return self.__mode

    @mode.setter
    def mode(self, m: Union[str, ReactionMode]):
        self.__mode = ReactionMode.get(m)

    def featurize(
        self,
        reaction: tuple[Chem.Mol, Chem.Mol],
        atom_features_extra: Optional[np.ndarray] = None,
        bond_features_extra: Optional[np.ndarray] = None,
    ) -> MolGraph:
        if atom_features_extra is not None:
            warnings.warn("'atom_features_extra' is currently unsupported for reactions")
        if bond_features_extra is not None:
            warnings.warn("'bond_features_extra' is currently unsupported for reactions")

        rct, pdt = reaction
        ri2pj, pids, rids = self.map_reac_to_prod(rct, pdt)

        X_v = self._calc_node_feature_matrix(rct, pdt, ri2pj, pids, rids)

        n_atoms = len(X_v)
        n_atoms_r = rct.GetNumAtoms()

        n_bonds = 0
        X_e = []
        a2b = [[] for _ in range(n_atoms)]
        b2a = []
        b2revb = []

        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                b_reac, b_prod = self._get_bonds(rct, pdt, ri2pj, pids, n_atoms_r, a1, a2)
                if b_reac is None and b_prod is None:
                    continue

                x_e = self._calc_edge_feature(b_reac, b_prod)
                if self.bond_messages:
                    X_e.append(np.hstack((X_v[a1], x_e)))
                    X_e.append(np.hstack((X_v[a2], x_e)))
                else:
                    X_e.append(x_e)
                    X_e.append(x_e)

                b12 = n_bonds
                b21 = b12 + 1
                a2b[a2].append(b12)
                a2b[a1].append(b21)
                b2a.extend([a1, a2])
                b2revb.extend([b21, b12])
                n_bonds += 2

        X_e = np.array(X_e)

        return MolGraph(n_atoms, n_bonds, X_v, X_e, a2b, b2a, b2revb, None, None)

    def _calc_node_feature_matrix(
        self, rct: Mol, pdt: Mol, ri2pj: dict[int, int], pids: Iterable[int], rids: Iterable[int]
    ) -> np.ndarray:
        """Calculate the global node feature matrix for the reaction"""
        X_v_r1 = np.array([self.atom_featurizer(a) for a in rct.GetAtoms()])
        X_v_p2 = np.array([self.atom_featurizer(pdt.GetAtomWithIdx(i)) for i in pids])
        X_v_p2 = X_v_p2.reshape(-1, X_v_r1.shape[1])

        if self.mode in [ReactionMode.REAC_DIFF, ReactionMode.PROD_DIFF, ReactionMode.REAC_PROD]:
            # Reactant:
            # (1) regular features for each atom in the reactants
            # (2) zero features for each atom that only in the products
            X_v_r2 = [self.atom_featurizer.featurize_num_only(pdt.GetAtomWithIdx(i)) for i in pids]
            X_v_r2 = np.array(X_v_r2).reshape(-1, X_v_r1.shape[1])

            # Product:
            # (1) either (a) product-side features for each atom in both
            #         or (b) zero features for each atom only in the reatants
            # (2) regular features for each atom only in the products
            X_v_p1 = np.array(
                [
                    self.atom_featurizer(pdt.GetAtomWithIdx(ri2pj[a.GetIdx()]))
                    if a.GetIdx() not in rids
                    else self.atom_featurizer.featurize_num_only(a)
                    for a in rct.GetAtoms()
                ]
            )
        else:
            # Reactant:
            # (1) regular features for each atom in the reactants
            # (2) regular features for each atom only in the products
            X_v_r2 = [self.atom_featurizer(pdt.GetAtomWithIdx(i)) for i in pids]
            X_v_r2 = np.array(X_v_r2).reshape(-1, X_v_r1.shape[1])

            # Product:
            # (1) either (a) product-side features for each atom in both
            #         or (b) reactant-side features for each atom only in the reatants
            # (2) regular features for each atom only in the products
            X_v_p1 = np.array(
                [
                    self.atom_featurizer(pdt.GetAtomWithIdx(ri2pj[a.GetIdx()]))
                    if a.GetIdx() not in rids
                    else self.atom_featurizer(a)
                    for a in rct.GetAtoms()
                ]
            )

        X_v_r = np.concatenate((X_v_r1, X_v_r2))
        X_v_p = np.concatenate((X_v_p1, X_v_p2))

        m = min(len(X_v_r), len(X_v_p))

        if self.mode in [ReactionMode.REAC_PROD, ReactionMode.REAC_PROD_BALANCE]:
            X_v = np.hstack((X_v_r[:m], X_v_p[:m, self.atom_featurizer.max_atomic_num + 1 :]))
        else:
            X_v_d = X_v_p[:m] - X_v_r[:m]
            if self.mode in [ReactionMode.REAC_DIFF, ReactionMode.REAC_DIFF_BALANCE]:
                X_v = np.hstack((X_v_r[:m], X_v_d[:m, self.atom_featurizer.max_atomic_num + 1 :]))
            else:
                X_v = np.hstack((X_v_p[:m], X_v_d[:m, self.atom_featurizer.max_atomic_num + 1 :]))

        return X_v

    def _get_bonds(
        self,
        rct: Bond,
        pdt: Bond,
        ri2pj: dict[int, int],
        pids: Sequence[int],
        n_atoms_r: int,
        a1: int,
        a2: int,
    ) -> tuple[Bond, Bond]:
        """get the reactant- and product-side bonds, respectively, betweeen atoms `a1` and `a2`"""
        if a1 >= n_atoms_r and a2 >= n_atoms_r:
            b_prod = pdt.GetBondBetweenAtoms(pids[a1 - n_atoms_r], pids[a2 - n_atoms_r])

            if self.mode in [
                ReactionMode.REAC_PROD_BALANCE,
                ReactionMode.REAC_DIFF_BALANCE,
                ReactionMode.PROD_DIFF_BALANCE,
            ]:
                b_reac = b_prod
            else:
                b_reac = None
        elif a1 < n_atoms_r and a2 >= n_atoms_r:  # One atom only in product
            b_reac = None

            if a1 in ri2pj:
                b_prod = pdt.GetBondBetweenAtoms(ri2pj[a1], pids[a2 - n_atoms_r])
            else:  # Atom atom only in reactant, the other only in product
                b_prod = None
        else:
            b_reac = rct.GetBondBetweenAtoms(a1, a2)

            if a1 in ri2pj and a2 in ri2pj:  # Both atoms in both reactant and product
                b_prod = pdt.GetBondBetweenAtoms(ri2pj[a1], ri2pj[a2])
            else:  # One or both atoms only in reactant
                if self.mode in [
                    ReactionMode.REAC_PROD_BALANCE,
                    ReactionMode.REAC_DIFF_BALANCE,
                    ReactionMode.PROD_DIFF_BALANCE,
                ]:
                    b_prod = None if (a1 in ri2pj or a2 in ri2pj) else b_reac
                else:
                    b_prod = None

        return b_reac, b_prod

    def _calc_edge_feature(self, b_reac: Bond, b_prod: Bond):
        """Calculate the global features of the two bonds"""
        x_e_r = self.bond_featurizer(b_reac)
        x_e_p = self.bond_featurizer(b_prod)

        if self.mode in [ReactionMode.REAC_PROD, ReactionMode.REAC_PROD_BALANCE]:
            x_e = np.hstack((x_e_r, x_e_p))
        else:
            x_e_d = x_e_p - x_e_r

            if self.mode in [ReactionMode.REAC_DIFF, ReactionMode.REAC_DIFF_BALANCE]:
                x_e_r = self.bond_featurizer(b_reac)
                x_e = np.hstack((x_e_r, x_e_d))
            else:
                x_e_p = self.bond_featurizer(b_prod)
                x_e = np.hstack((x_e_p, x_e_d))

        return x_e

    @staticmethod
    def map_reac_to_prod(
        reactants: Chem.Mol, products: Chem.Mol
    ) -> tuple[dict[int, int], list[int], list[int]]:
        """Map atom indices between corresponding atoms in the reactant and product molecules

        Parameters
        ----------
        reactants : Chem.Mol
            An RDKit molecule of the reactants
        products : Chem.Mol
            An RDKit molecule of the products

        Returns
        -------
        ri2pi : dict[int, int]
            A dictionary of corresponding atom indices from reactant atoms to product atoms
        pdt_idxs : list[int]
            atom indices of poduct atoms
        rct_idxs : list[int]
            atom indices of reactant atoms
        """
        pdt_idxs = []
        mapno2pj = {}
        mapnos_reac = {a.GetAtomMapNum() for a in reactants.GetAtoms()}

        for a in products.GetAtoms():
            mapno = a.GetAtomMapNum()
            pj = a.GetIdx()

            if mapno > 0:
                mapno2pj[mapno] = pj
                if mapno not in mapnos_reac:
                    pdt_idxs.append(pj)
            else:
                pdt_idxs.append(pj)

        rct_idxs = []
        ri2pj = {}

        for a in reactants.GetAtoms():
            mapno = a.GetAtomMapNum()
            ri = a.GetIdx()

            if mapno > 0:
                try:
                    ri2pj[ri] = mapno2pj[mapno]
                except KeyError:
                    rct_idxs.append(ri)
            else:
                rct_idxs.append(ri)

        return ri2pj, pdt_idxs, rct_idxs
