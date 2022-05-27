from __future__ import annotations

from enum import Enum, auto
from typing import Optional
import warnings
import numpy as np
from rdkit import Chem

from chemprop.featurizers.v2.molgraph import MolGraph
from chemprop.featurizers.v2.multihot import AtomFeaturizer, BondFeaturizer


class ReactionMode(Enum):
    """The manner in which a reaction should be featurized into a `MolGraph`

    REAC_PROD
        concatenates the reactants feature with the products feature.
    REAC_PROD_BALANCE
        concatenates the reactants feature with the products feature, balances imbalanced reactions.
    REAC_DIFF
        concatenates the reactants feature with the difference in features between reactants and
        products.
    REAC_DIFF_BALANCE
        concatenates the reactants feature with the difference in features between reactants and
        products, balances imbalanced reactions.
    PROD_DIFF
        concatenates the products feature with the difference in features between reactants and
        products.
    PROD_DIFF_BALANCE
        concatenates the products feature with the difference in features between reactants and
        products, balances imbalanced reactions.
    """

    REAC_PROD = auto()
    REAC_PROD_BALANCE = auto()
    REAC_DIFF = auto()
    REAC_DIFF_BALANCE = auto()
    PROD_DIFF = auto()
    PROD_DIFF_BALANCE = auto()


def map_reac_to_prod(
    mol_reac: Chem.Mol, mol_prod: Chem.Mol
) -> tuple[dict[int, int], list[int], list[int]]:
    """Map atom indices between corresponding atoms in the reactant and product molecules

    Parameters
    ----------
    mol_reac
        An RDKit molecule of the reactants
    mol_prod
        An RDKit molecule of the products

    Returns
    -------
    r2p : dict[int, int]
        A dictionary of corresponding atom indices from reactant atoms to product atoms
    pids : list[int]
        atom indices of poduct atoms
    rids : [int]
        atom indices of reactant atoms
    """
    pids = []
    prod_map_to_id = {}
    mapnos_reac = {a.GetAtomMapNum() for a in mol_reac.GetAtoms()}

    for a in mol_prod.GetAtoms():
        mapno = a.GetAtomMapNum()
        i = a.GetIdx()

        if mapno > 0:
            prod_map_to_id[mapno] = i
            if mapno not in mapnos_reac:
                pids.append(i)
        else:
            pids.append(i)

    rids = []
    r2p = {}

    for a in mol_reac.GetAtoms():
        mapno = a.GetAtomMapNum()
        i = a.GetIdx()

        if mapno > 0:
            try:
                r2p[i] = prod_map_to_id[mapno]
            except KeyError:
                rids.append(i)
        else:
            rids.append(i)

    return r2p, pids, rids


class ReactionFeaturizer:
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
        atom_feautrizer: Optional[AtomFeaturizer] = None,
        bond_featurizer: Optional[BondFeaturizer] = None,
        atom_messages: bool = False,
    ):
        self.mode = mode
        self.atom_featurizer = atom_feautrizer or AtomFeaturizer()
        self.bond_feautrizer = bond_featurizer or BondFeaturizer()
        self.atom_messages = atom_messages

    def featurize(
        self,
        reaction: tuple[Chem.Mol],
        atom_features_extra: Optional[np.ndarray] = None,
        bond_features_extra: Optional[np.ndarray] = None,
    ):
        """Featurize the input reaction into a molecular graph

        Parameters
        ----------
        reaction : tuple[Chem.Mol]
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
        n_atoms = 0
        n_bonds = 0
        X_v = []
        X_e = []
        a2b = []
        b2a = []
        b2revb = []

        if atom_features_extra is not None:
            warnings.warn("Extra atom features are currently not supported for reactions")
        if bond_features_extra is not None:
            warnings.warn("Extra bond features are currently not supported for reactions")

        reactant = reaction[0]
        product = reaction[1]
        ri2pi, pids, rids = map_reac_to_prod(reactant, product)

        if self.reaction_mode in [
            ReactionMode.REAC_DIFF,
            ReactionMode.PROD_DIFF,
            ReactionMode.REAC_PROD,
        ]:
            # Reactant: regular atom features for each atom in the reactants, as well as zero features for atoms that are only in the products (indices in pio)
            X_v_r = [self.atom_featurizer(a) for a in reactant.GetAtoms()] + [
                self.atom_featurizer.featurize_num_only(product.GetAtomWithIdx(i)) for i in pids
            ]

            # Product: regular atom features for each atom that is in both reactants and products
            # (not in rio), other atom features zero,
            # regular features for atoms that are only in the products (indices in pio)
            X_v_p = [
                self.atom_featurizer(product.GetAtomWithIdx(ri2pi[a.GetIdx()]))
                if a.GetIdx() not in rids
                else self.atom_featurizer.featurize_num_only(a)
                for a in reactant.GetAtoms()
            ] + [self.atom_featurizer(product.GetAtomWithIdx(i)) for i in pids]
        else:  # balance
            # Reactant: regular atom features for each atom in the reactants, copy features from
            # product side for atoms that are only in the products (indices in pio)
            # X_v_r = (
            #     [self.atom_featurizer(a) for a in reactant.GetAtoms()]
            #     + [self.atom_featurizer(product.GetAtomWithIdx(i)) for i in pio]
            # )
            X_v_r = np.hstack(
                (
                    np.array([self.atom_featurizer(a) for a in reactant.GetAtoms()]),
                    np.array([self.atom_featurizer(product.GetAtomWithIdx(i)) for i in pids]),
                )
            )

            # Product: regular atom features for each atom that is in both reactants and products
            # (not in rio), copy features from reactant side for
            # other atoms, regular features for atoms that are only in the products (indices in pio)
            X_v_p = np.hstack(
                (
                    np.array(
                        [
                            self.atom_featurizer(product.GetAtomWithIdx(ri2pi[a.GetIdx()]))
                            if a.GetIdx() not in rids
                            else self.atom_featurizer(a)
                            for a in reactant.GetAtoms()
                        ]
                    ),
                    np.array([self.atom_featurizer(product.GetAtomWithIdx(i)) for i in pids]),
                )
            )

        m = min(len(X_v_p), len(X_v_d))

        if self.reaction_mode in [
            ReactionMode.REAC_DIFF,
            ReactionMode.REAC_DIFF_BALANCE,
            ReactionMode.PROD_DIFF,
            ReactionMode.PROD_DIFF_BALANCE,
        ]:
            X_v_d = [
                list(map(lambda x, y: x - y, x_v_p, x_v_r)) for x_v_p, x_v_r in zip(X_v_p, X_v_r)
            ]
            # X_v_d = X_v_p[:m] - X_v_r[:m]

        if self.reaction_mode in [ReactionMode.REAC_PROD, ReactionMode.REAC_PROD_BALANCE]:
            X_v = [x + y[self.atom_featurizer.max_atomic_num + 1 :] for x, y in zip(X_v_r, X_v_p)]
            # X_v = np.hstack((X_v_r[:m], X_v_p[:m, self.atom_featurizer.max_atomic_num + 1 :]))

        elif self.reaction_mode in [ReactionMode.REAC_DIFF, ReactionMode.REAC_DIFF_BALANCE]:
            X_v = [x + y[self.atom_featurizer.max_atomic_num + 1 :] for x, y in zip(X_v_r, X_v_d)]
            # X_v = np.hstack((X_v_r[:m], X_v_d[:m, self.atom_featurizer.max_atomic_num + 1 :]))
        elif self.reaction_mode in [ReactionMode.PROD_DIFF, ReactionMode.PROD_DIFF_BALANCE]:
            X_v = [x + y[self.atom_featurizer.max_atomic_num + 1 :] for x, y in zip(X_v_p, X_v_d)]
            # X_v = np.hstack((X_v_p[:m], X_v_d[:m, self.atom_featurizer.max_atomic_num + 1 :]))

        n_atoms = len(X_v)
        n_atoms_reac = reactant.GetNumAtoms()

        a2b = [[] for _ in range(n_atoms)]

        i = 0
        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                if a1 >= n_atoms_reac and a2 >= n_atoms_reac:
                    # Both atoms only in product
                    bond_prod = product.GetBondBetweenAtoms(
                        pids[a1 - n_atoms_reac], pids[a2 - n_atoms_reac]
                    )

                    if self.reaction_mode in [
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
                        if self.reaction_mode in [
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

                if self.reaction_mode in [ReactionMode.REAC_PROD, ReactionMode.REAC_PROD_BALANCE]:
                    x_e_r = self.bond_featurizer(bond_reac)
                    x_e_p = self.bond_featurizer(bond_prod)
                    x_e = np.hstack((x_e_r, x_e_p))
                else:
                    # x_e_d = [y - x for x, y in zip(x_e_r, x_e_p)]
                    x_e_d = x_e_p - x_e_r

                    if self.reaction_mode in [
                        ReactionMode.REAC_DIFF,
                        ReactionMode.REAC_DIFF_BALANCE,
                    ]:
                        x_e_r = self.bond_featurizer(bond_reac)
                        x_e = np.hstack((x_e_r, x_e_d))
                    else:
                        x_e_p = self.bond_featurizer(bond_prod)
                        x_e = np.hstack((x_e_p, x_e_d))

                X_e.append(np.hstack((X_v[a1], x_e)))
                X_e.append(np.hstack((X_v[a2], x_e)))

                b12 = i
                b21 = b12 + 1
                a2b[a2].append(b12)
                b2a.append(a1)
                a2b[a1].append(b21)
                b2a.append(a2)
                b2revb.append(b21)
                b2revb.append(b12)

                # b2a.extend([a1, a2])
                # b2revb.extend([b21, b12])
                i += 2

        return MolGraph(n_atoms, 2 * n_bonds, X_v, X_e, a2b, b2a, b2revb, None, None)
