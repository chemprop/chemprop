from __future__ import annotations

from enum import Enum, auto
from typing import Optional
import warnings
import numpy as np
from rdkit import Chem
from chemprop.featurizers.v2.base import MolGraphFeaturizer

from chemprop.featurizers.v2.molgraph import MolGraph
from chemprop.featurizers.v2.multihot import AtomFeaturizer, BondFeaturizer


class ReactionMode(Enum):
    """The manner in which a reaction should be featurized into a `MolGraph`

    REAC_PROD
        concatenate the reactant features with the product features.
    REAC_PROD_BALANCE
        concatenate the reactant features with the products feature and balances imbalanced
        reactions.
    REAC_DIFF
        concatenates the reactant features with the difference in features between reactants and
        products
    REAC_DIFF_BALANCE
        concatenates the reactant features with the difference in features between reactants and
        products and balances imbalanced reactions
    PROD_DIFF
        concatenates the product features with the difference in features between reactants and
        products
    PROD_DIFF_BALANCE
        concatenates the product features with the difference in features between reactants and
        products and balances imbalanced reactions
    """

    REAC_PROD = auto()
    REAC_PROD_BALANCE = auto()
    REAC_DIFF = auto()
    REAC_DIFF_BALANCE = auto()
    PROD_DIFF = auto()
    PROD_DIFF_BALANCE = auto()


def map_reac_to_prod(
    reactants: Chem.Mol, products: Chem.Mol
) -> tuple[dict[int, int], list[int], list[int]]:
    """Map atom indices between corresponding atoms in the reactant and product molecules

    Parameters
    ----------
    reactants
        An RDKit molecule of the reactants
    products
        An RDKit molecule of the products

    Returns
    -------
    r2p : dict[int, int]
        A dictionary of corresponding atom indices from reactant atoms to product atoms
    pdt_idxs : list[int]
        atom indices of poduct atoms
    rct_idxs : list[int]
        atom indices of reactant atoms
    """
    pdt_idxs = []
    prod_map_to_id = {}
    mapnos_reac = {a.GetAtomMapNum() for a in reactants.GetAtoms()}

    for a in products.GetAtoms():
        mapno = a.GetAtomMapNum()
        i = a.GetIdx()

        if mapno > 0:
            prod_map_to_id[mapno] = i
            if mapno not in mapnos_reac:
                pdt_idxs.append(i)
        else:
            pdt_idxs.append(i)

    rct_idxs = []
    r2p = {}

    for a in reactants.GetAtoms():
        mapno = a.GetAtomMapNum()
        i = a.GetIdx()

        if mapno > 0:
            try:
                r2p[i] = prod_map_to_id[mapno]
            except KeyError:
                rct_idxs.append(i)
        else:
            rct_idxs.append(i)

    return r2p, pdt_idxs, rct_idxs


class ReactionFeaturizer(MolGraphFeaturizer):
    """A `ReactionFeaturizer` featurizes reactions into `MolGraph`s

    Attributes
    ----------
    atom_featurizer : AtomFeaturizer
    bond_featurizer : BondFeaturizer
    atom_fdim : int
        the dimension of atom feature represenatations in this featurizer
    bond_fdim : int
        the dimension of bond feature represenatations in this featurizer
    atom_messages : bool

    Parameters
    ----------
    mode : ReactionMode
        the mode by which to featurize the reaction
    atom_featurizer : AtomFeaturizer, default=AtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=BondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    atom_messages : bool, default=False
        whether to prepare the `MolGraph` for use with atom-based messages
    """

    def __init__(
        self,
        mode: ReactionMode,
        atom_featurizer: Optional[AtomFeaturizer] = None,
        bond_featurizer: Optional[BondFeaturizer] = None,
        atom_messages: bool = False,
    ):
        super().__init__(atom_featurizer, bond_featurizer, atom_messages)

        self.mode = mode
        if not self.atom_messages:
            self.bond_fdim += self.atom_fdim

    def featurize(
        self,
        reaction: tuple[Chem.Mol, Chem.Mol],
        atom_features_extra: Optional[np.ndarray] = None,
        bond_features_extra: Optional[np.ndarray] = None,
    ):
        """Featurize the input reaction into a molecular graph

        Parameters
        ----------
        reaction : tuple[Chem.Mol, Chem.Mol]
            a 2-tuple of atom-mapped rdkit molecules, where the 0th element is the reactant and the
            1st element is the product
        atom_features_extra : Optional[np.ndarray], default=None
            *UNSUPPORTED* maintained only to maintain parity with the method signature of the
            `MoleculeFeaturizer`
        bond_features_extra : Optional[np.ndarray], default=None
            *UNSUPPORTED* maintained only to maintain parity with the method signature of the
            `MoleculeFeaturizer`

        Returns
        -------
        MolGraph
            the molecular graph of the input reaction
        """
        if atom_features_extra is not None:
            warnings.warn("Extra atom features are currently not supported for reactions")
        if bond_features_extra is not None:
            warnings.warn("Extra bond features are currently not supported for reactions")

        reactant, product = reaction
        ri2pi, pids, rids = map_reac_to_prod(reactant, product)

        if self.mode in [ReactionMode.REAC_DIFF, ReactionMode.PROD_DIFF, ReactionMode.REAC_PROD]:
            # Reactant: regular atom features for each atom in the reactants, as well as zero
            # features for atoms that are only in the products (indices in pio)
            A = np.array([self.atom_featurizer(a) for a in reactant.GetAtoms()])
            B = np.array(
                [self.atom_featurizer.featurize_num_only(product.GetAtomWithIdx(i)) for i in pids]
            ).reshape(-1, self.atom_fdim)
            X_v_r = np.concatenate((A, B))

            # Product: regular atom features for each atom that is in both reactants and products
            # (not in rio), other atom features zero,
            # regular features for atoms that are only in the products (indices in pio)
            C = np.array(
                [
                    self.atom_featurizer(product.GetAtomWithIdx(ri2pi[a.GetIdx()]))
                    if a.GetIdx() not in rids
                    else self.atom_featurizer.featurize_num_only(a)
                    for a in reactant.GetAtoms()
                ]
            )
            D = np.array([self.atom_featurizer(product.GetAtomWithIdx(i)) for i in pids])
            D = D.reshape(-1, self.atom_fdim)
            X_v_p = np.concatenate((C, D))
        else:  # balance
            # Reactant: regular atom features for each atom in the reactants, copy features from
            #   product side for atoms that are only in the products (:= indices in pio)
            A = np.array([self.atom_featurizer(a) for a in reactant.GetAtoms()])
            B = np.array([self.atom_featurizer(product.GetAtomWithIdx(i)) for i in pids]).reshape(
                -1, self.atom_fdim
            )
            X_v_r = np.concatenate((A, B))

            # Product: (1) regular atom features for each atom that is in both reactants and
            #   products (:= indices not in rids), copy features from reactant side for other
            #   atoms, (2) regular features for atoms that are only in the products (:= indices in
            #   pio)
            C = np.array(
                [
                    self.atom_featurizer(product.GetAtomWithIdx(ri2pi[a.GetIdx()]))
                    if a.GetIdx() not in rids
                    else self.atom_featurizer(a)
                    for a in reactant.GetAtoms()
                ]
            )
            D = np.array([self.atom_featurizer(product.GetAtomWithIdx(i)) for i in pids])
            D = D.reshape(-1, self.atom_fdim)
            X_v_p = np.concatenate((C, D))

        m = min(len(X_v_r), len(X_v_p))

        if self.mode in [
            ReactionMode.REAC_DIFF,
            ReactionMode.REAC_DIFF_BALANCE,
            ReactionMode.PROD_DIFF,
            ReactionMode.PROD_DIFF_BALANCE,
        ]:
            X_v_d = X_v_p[:m] - X_v_r[:m]

        if self.mode in [ReactionMode.REAC_PROD, ReactionMode.REAC_PROD_BALANCE]:
            X_v = np.hstack((X_v_r[:m], X_v_p[:m, self.atom_featurizer.max_atomic_num + 1 :]))
        elif self.mode in [ReactionMode.REAC_DIFF, ReactionMode.REAC_DIFF_BALANCE]:
            X_v = np.hstack((X_v_r[:m], X_v_d[:m, self.atom_featurizer.max_atomic_num + 1 :]))
        elif self.mode in [ReactionMode.PROD_DIFF, ReactionMode.PROD_DIFF_BALANCE]:
            X_v = np.hstack((X_v_p[:m], X_v_d[:m, self.atom_featurizer.max_atomic_num + 1 :]))

        n_atoms = len(X_v)
        n_atoms_reac = reactant.GetNumAtoms()

        n_bonds = 0
        X_e = []
        a2b = [[] for _ in range(n_atoms)]
        b2a = []
        b2revb = []

        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                if a1 >= n_atoms_reac and a2 >= n_atoms_reac:
                    # Both atoms only in product
                    bond_prod = product.GetBondBetweenAtoms(
                        pids[a1 - n_atoms_reac], pids[a2 - n_atoms_reac]
                    )

                    if self.mode in [
                        ReactionMode.REAC_PROD_BALANCE,
                        ReactionMode.REAC_DIFF_BALANCE,
                        ReactionMode.PROD_DIFF_BALANCE,
                    ]:
                        bond_reac = bond_prod
                    else:
                        bond_reac = None
                elif a1 < n_atoms_reac and a2 >= n_atoms_reac:
                    # One atom only in product
                    bond_reac = None

                    if a1 in ri2pi.keys():
                        bond_prod = product.GetBondBetweenAtoms(ri2pi[a1], pids[a2 - n_atoms_reac])
                    else:
                        # Atom atom only in reactant, the other only in product
                        bond_prod = None
                else:
                    bond_reac = reactant.GetBondBetweenAtoms(a1, a2)

                    if a1 in ri2pi.keys() and a2 in ri2pi.keys():
                        # Both atoms in both reactant and product
                        bond_prod = product.GetBondBetweenAtoms(ri2pi[a1], ri2pi[a2])
                    else:
                        if self.mode in [
                            ReactionMode.REAC_PROD_BALANCE,
                            ReactionMode.REAC_DIFF_BALANCE,
                            ReactionMode.PROD_DIFF_BALANCE,
                        ]:
                            if a1 in ri2pi.keys() or a2 in ri2pi.keys():
                                # One atom only in reactant
                                bond_prod = None
                            else:
                                # Both atoms only in reactant
                                bond_prod = bond_reac
                        else:
                            # One or both atoms only in reactant
                            bond_prod = None

                if bond_reac is None and bond_prod is None:
                    continue

                x_e_r = self.bond_featurizer(bond_reac)
                x_e_p = self.bond_featurizer(bond_prod)

                if self.mode in [ReactionMode.REAC_PROD, ReactionMode.REAC_PROD_BALANCE]:
                    x_e = np.hstack((x_e_r, x_e_p))
                else:
                    x_e_d = x_e_p - x_e_r

                    if self.mode in [ReactionMode.REAC_DIFF, ReactionMode.REAC_DIFF_BALANCE]:
                        x_e_r = self.bond_featurizer(bond_reac)
                        x_e = np.hstack((x_e_r, x_e_d))
                    else:
                        x_e_p = self.bond_featurizer(bond_prod)
                        x_e = np.hstack((x_e_p, x_e_d))

                if self.atom_messages:
                    X_e.append(x_e)
                    X_e.append(x_e)
                else:
                    X_e.append(np.hstack((X_v[a1], x_e)))
                    X_e.append(np.hstack((X_v[a2], x_e)))

                b12 = n_bonds
                b21 = b12 + 1
                a2b[a2].append(b12)
                b2a.append(a1)
                a2b[a1].append(b21)
                b2a.append(a2)
                b2revb.append(b21)
                b2revb.append(b12)

                # b2a.extend([a1, a2])
                # b2revb.extend([b21, b12])
                n_bonds += 2
        X_e = np.array(X_e)

        return MolGraph(n_atoms, n_bonds, X_v, X_e, a2b, b2a, b2revb, None, None)
