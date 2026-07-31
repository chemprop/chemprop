from dataclasses import InitVar, dataclass, field
from enum import auto
import logging
from typing import Iterable, Literal, Sequence, TypeAlias

import cuik_molmaker
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Bond, Mol
import torch

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import GraphFeaturizer
from chemprop.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin
from chemprop.featurizers.molgraph.molecule import BatchCuikMolGraph
from chemprop.types import Rxn
from chemprop.utils.utils import EnumMapping

logger = logging.getLogger(__name__)


class RxnMode(EnumMapping):
    """The mode by which a reaction should be featurized into a `MolGraph`"""

    REAC_PROD = auto()
    """concatenate the reactant features with the product features."""
    REAC_PROD_BALANCE = auto()
    """concatenate the reactant features with the products feature and balances imbalanced
    reactions"""
    REAC_DIFF = auto()
    """concatenates the reactant features with the difference in features between reactants and
    products"""
    REAC_DIFF_BALANCE = auto()
    """concatenates the reactant features with the difference in features between reactants and
    product and balances imbalanced reactions"""
    PROD_DIFF = auto()
    """concatenates the product features with the difference in features between reactants and
    products"""
    PROD_DIFF_BALANCE = auto()
    """concatenates the product features with the difference in features between reactants and
    products and balances imbalanced reactions"""


@dataclass
class CondensedGraphOfReactionFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Rxn]):
    """A :class:`CondensedGraphOfReactionFeaturizer` featurizes reactions using the condensed
    reaction graph method utilized in [1]_

    **NOTE**: This class *does not* accept a :class:`AtomFeaturizer` instance. This is because
    it requries the :meth:`num_only()` method, which is only implemented in the concrete
    :class:`AtomFeaturizer` class

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=AtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizerBase, default=BondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    mode_ : Union[str, ReactionMode], default=ReactionMode.REAC_DIFF
        the mode by which to featurize the reaction as either the string code or enum value

    References
    ----------
    .. [1] Heid, E.; Green, W.H. "Machine Learning of Reaction Properties via Learned
        Representations of the Condensed Graph of Reaction." J. Chem. Inf. Model. 2022, 62,
        2101-2110. https://doi.org/10.1021/acs.jcim.1c00975
    """

    mode_: InitVar[str | RxnMode] = RxnMode.REAC_DIFF

    def __post_init__(self, mode_: str | RxnMode):
        super().__post_init__()

        self.mode = mode_
        self.atom_fdim += len(self.atom_featurizer) - len(self.atom_featurizer.atomic_nums) - 1
        self.bond_fdim *= 2

    @property
    def mode(self) -> RxnMode:
        return self.__mode

    @mode.setter
    def mode(self, m: str | RxnMode):
        self.__mode = RxnMode.get(m)

    def __call__(
        self,
        rxn: tuple[Chem.Mol, Chem.Mol],
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        """Featurize the input reaction into a molecular graph

        Parameters
        ----------
        rxn : Rxn
            a 2-tuple of atom-mapped rdkit molecules, where the 0th element is the reactant and the
            1st element is the product
        atom_features_extra : np.ndarray | None, default=None
            *UNSUPPORTED* maintained only to maintain parity with the method signature of the
            `MoleculeFeaturizer`
        bond_features_extra : np.ndarray | None, default=None
            *UNSUPPORTED* maintained only to maintain parity with the method signature of the
            `MoleculeFeaturizer`

        Returns
        -------
        MolGraph
            the molecular graph of the reaction
        """

        if atom_features_extra is not None:
            logger.warning("'atom_features_extra' is currently unsupported for reactions")
        if bond_features_extra is not None:
            logger.warning("'bond_features_extra' is currently unsupported for reactions")

        reac, pdt = rxn
        r2p_idx_map, pdt_idxs, reac_idxs = self.map_reac_to_prod(reac, pdt)

        V = self._calc_node_feature_matrix(reac, pdt, r2p_idx_map, pdt_idxs, reac_idxs)
        E = []
        edge_index = [[], []]

        n_atoms_tot = len(V)
        n_atoms_reac = reac.GetNumAtoms()

        for u in range(n_atoms_tot):
            for v in range(u + 1, n_atoms_tot):
                b_reac, b_prod = self._get_bonds(
                    reac, pdt, r2p_idx_map, pdt_idxs, n_atoms_reac, u, v
                )
                if b_reac is None and b_prod is None:
                    continue

                x_e = self._calc_edge_feature(b_reac, b_prod)
                E.extend([x_e, x_e])
                edge_index[0].extend([u, v])
                edge_index[1].extend([v, u])

        E = np.array(E) if len(E) > 0 else np.empty((0, self.bond_fdim))
        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return MolGraph(V, E, edge_index, rev_edge_index)

    def _calc_node_feature_matrix(
        self,
        rct: Mol,
        pdt: Mol,
        r2p_idx_map: dict[int, int],
        pdt_idxs: Iterable[int],
        reac_idxs: Iterable[int],
    ) -> np.ndarray:
        """Calculate the node feature matrix for the reaction"""
        X_v_r1 = np.array([self.atom_featurizer(a) for a in rct.GetAtoms()])
        X_v_p2 = np.array([self.atom_featurizer(pdt.GetAtomWithIdx(i)) for i in pdt_idxs])
        X_v_p2 = X_v_p2.reshape(-1, X_v_r1.shape[1])

        if self.mode in [RxnMode.REAC_DIFF, RxnMode.PROD_DIFF, RxnMode.REAC_PROD]:
            # Reactant:
            # (1) regular features for each atom in the reactants
            # (2) zero features for each atom that's only in the products
            X_v_r2 = [self.atom_featurizer.num_only(pdt.GetAtomWithIdx(i)) for i in pdt_idxs]
            X_v_r2 = np.array(X_v_r2).reshape(-1, X_v_r1.shape[1])

            # Product:
            # (1) either (a) product-side features for each atom in both
            #         or (b) zero features for each atom only in the reatants
            # (2) regular features for each atom only in the products
            X_v_p1 = np.array(
                [
                    (
                        self.atom_featurizer(pdt.GetAtomWithIdx(r2p_idx_map[a.GetIdx()]))
                        if a.GetIdx() not in reac_idxs
                        else self.atom_featurizer.num_only(a)
                    )
                    for a in rct.GetAtoms()
                ]
            )
        else:
            # Reactant:
            # (1) regular features for each atom in the reactants
            # (2) regular features for each atom only in the products
            X_v_r2 = [self.atom_featurizer(pdt.GetAtomWithIdx(i)) for i in pdt_idxs]
            X_v_r2 = np.array(X_v_r2).reshape(-1, X_v_r1.shape[1])

            # Product:
            # (1) either (a) product-side features for each atom in both
            #         or (b) reactant-side features for each atom only in the reatants
            # (2) regular features for each atom only in the products
            X_v_p1 = np.array(
                [
                    (
                        self.atom_featurizer(pdt.GetAtomWithIdx(r2p_idx_map[a.GetIdx()]))
                        if a.GetIdx() not in reac_idxs
                        else self.atom_featurizer(a)
                    )
                    for a in rct.GetAtoms()
                ]
            )

        X_v_r = np.concatenate((X_v_r1, X_v_r2))
        X_v_p = np.concatenate((X_v_p1, X_v_p2))

        m = min(len(X_v_r), len(X_v_p))

        if self.mode in [RxnMode.REAC_PROD, RxnMode.REAC_PROD_BALANCE]:
            X_v = np.hstack((X_v_r[:m], X_v_p[:m, len(self.atom_featurizer.atomic_nums) + 1 :]))
        else:
            X_v_d = X_v_p[:m] - X_v_r[:m]
            if self.mode in [RxnMode.REAC_DIFF, RxnMode.REAC_DIFF_BALANCE]:
                X_v = np.hstack((X_v_r[:m], X_v_d[:m, len(self.atom_featurizer.atomic_nums) + 1 :]))
            else:
                X_v = np.hstack((X_v_p[:m], X_v_d[:m, len(self.atom_featurizer.atomic_nums) + 1 :]))

        return X_v

    def _get_bonds(
        self,
        rct: Bond,
        pdt: Bond,
        ri2pj: dict[int, int],
        pids: Sequence[int],
        n_atoms_r: int,
        u: int,
        v: int,
    ) -> tuple[Bond, Bond]:
        """get the corresponding reactant- and product-side bond, respectively, betweeen atoms `u` and `v`"""
        if u >= n_atoms_r and v >= n_atoms_r:
            b_prod = pdt.GetBondBetweenAtoms(pids[u - n_atoms_r], pids[v - n_atoms_r])

            if self.mode in [
                RxnMode.REAC_PROD_BALANCE,
                RxnMode.REAC_DIFF_BALANCE,
                RxnMode.PROD_DIFF_BALANCE,
            ]:
                b_reac = b_prod
            else:
                b_reac = None
        elif u < n_atoms_r and v >= n_atoms_r:  # One atom only in product
            b_reac = None

            if u in ri2pj:
                b_prod = pdt.GetBondBetweenAtoms(ri2pj[u], pids[v - n_atoms_r])
            else:  # Atom atom only in reactant, the other only in product
                b_prod = None
        else:
            b_reac = rct.GetBondBetweenAtoms(u, v)

            if u in ri2pj and v in ri2pj:  # Both atoms in both reactant and product
                b_prod = pdt.GetBondBetweenAtoms(ri2pj[u], ri2pj[v])
            elif self.mode in [
                RxnMode.REAC_PROD_BALANCE,
                RxnMode.REAC_DIFF_BALANCE,
                RxnMode.PROD_DIFF_BALANCE,
            ]:
                b_prod = None if (u in ri2pj or v in ri2pj) else b_reac
            else:  # One or both atoms only in reactant
                b_prod = None

        return b_reac, b_prod

    def _calc_edge_feature(self, b_reac: Bond, b_pdt: Bond):
        """Calculate the global features of the two bonds"""
        x_e_r = self.bond_featurizer(b_reac)
        x_e_p = self.bond_featurizer(b_pdt)
        x_e_d = x_e_p - x_e_r

        if self.mode in [RxnMode.REAC_PROD, RxnMode.REAC_PROD_BALANCE]:
            x_e = np.hstack((x_e_r, x_e_p))
        elif self.mode in [RxnMode.REAC_DIFF, RxnMode.REAC_DIFF_BALANCE]:
            x_e = np.hstack((x_e_r, x_e_d))
        else:
            x_e = np.hstack((x_e_p, x_e_d))

        return x_e

    @classmethod
    def map_reac_to_prod(
        cls, reacs: Chem.Mol, pdts: Chem.Mol
    ) -> tuple[dict[int, int], list[int], list[int]]:
        """Map atom indices between corresponding atoms in the reactant and product molecules

        Parameters
        ----------
        reacs : Chem.Mol
            An RDKit molecule of the reactants
        pdts : Chem.Mol
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
        reac_atommap_nums = {a.GetAtomMapNum() for a in reacs.GetAtoms()}

        for a in pdts.GetAtoms():
            map_num = a.GetAtomMapNum()
            j = a.GetIdx()

            if map_num > 0:
                mapno2pj[map_num] = j
                if map_num not in reac_atommap_nums:
                    pdt_idxs.append(j)
            else:
                pdt_idxs.append(j)

        rct_idxs = []
        r2p_idx_map = {}

        for a in reacs.GetAtoms():
            map_num = a.GetAtomMapNum()
            i = a.GetIdx()

            if map_num > 0:
                try:
                    r2p_idx_map[i] = mapno2pj[map_num]
                except KeyError:
                    rct_idxs.append(i)
            else:
                rct_idxs.append(i)

        return r2p_idx_map, pdt_idxs, rct_idxs


CGRFeaturizer: TypeAlias = CondensedGraphOfReactionFeaturizer


@dataclass
class CuikmolmakerCGRFeaturizer:
    """A :class:`CuikmolmakerCGRFeaturizer` featurizes a batch of reactions using the Condensed
    Graph of Reaction (CGR) representation via the cuik-molmaker C++ library.

    Parameters
    ----------
    atom_featurizer_mode : str, default="V2"
        Atom featurizer mode: V1, V2, ORGANIC, or RIGR.
    reaction_mode : str, default="REAC_DIFF"
        CGR mode: REAC_DIFF, REAC_PROD, PROD_DIFF, REAC_DIFF_BALANCE, REAC_PROD_BALANCE, or
        PROD_DIFF_BALANCE.
    keep_h : bool, default=True
        Whether to keep explicit hydrogens when parsing SMILES. Required for reactions with mapped
        hydrogens.
    add_h : bool, default=False
        Whether to add implicit hydrogens via Chem.AddHs after parsing.
    """

    atom_featurizer_mode: Literal["V1", "V2", "ORGANIC", "RIGR"] = "V2"
    reaction_mode: Literal[
        "REAC_DIFF",
        "REAC_PROD",
        "PROD_DIFF",
        "REAC_DIFF_BALANCE",
        "REAC_PROD_BALANCE",
        "PROD_DIFF_BALANCE",
    ] = "REAC_DIFF"
    keep_h: bool = True
    add_h: bool = False

    atom_fdim: int = field(init=False)
    bond_fdim: int = field(init=False)

    def __post_init__(self):
        if not hasattr(cuik_molmaker, "batch_reaction_featurizer"):
            raise ImportError(
                "CuikmolmakerCGRFeaturizer requires cuik_molmaker>=0.3 for reaction (CGR) "
                "featurization, but the installed cuik_molmaker does not provide it. Please "
                "upgrade with `conda install 'conda-forge::cuik_molmaker>=0.3'` or "
                "`pip install --upgrade cuik_molmaker_pin`."
            )

        atom_props_float = ["aromatic", "mass"]
        bond_props = ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"]
        single_bond_fdim = 14

        self.atom_featurizer_mode = self.atom_featurizer_mode.upper()
        if self.atom_featurizer_mode == "V1":
            atom_props_onehot = [
                "atomic-number",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization",
            ]
            single_atom_fdim = 133
            atomic_num_block_w = 101  # 100 elements + 1 "other"
        elif self.atom_featurizer_mode == "V2":
            atom_props_onehot = [
                "atomic-number-common",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization-expanded",
            ]
            single_atom_fdim = 72
            atomic_num_block_w = 38  # 37 common elements + 1 "other"
        elif self.atom_featurizer_mode == "ORGANIC":
            atom_props_onehot = [
                "atomic-number-organic",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization-organic",
            ]
            single_atom_fdim = 44
            atomic_num_block_w = 13  # 12 organic elements + 1 "other"
        elif self.atom_featurizer_mode == "RIGR":
            atom_props_onehot = ["atomic-number-common", "total-degree", "num-hydrogens"]
            atom_props_float = ["mass"]
            bond_props = ["is-null", "in-ring"]
            single_atom_fdim = 52
            atomic_num_block_w = 38  # same as V2
            single_bond_fdim = 2
        else:
            raise ValueError(f"Invalid atom featurizer mode: {self.atom_featurizer_mode}")

        # CGR doubles the feature dimension; second half strips the atomic-number one-hot block
        self.atom_fdim = 2 * single_atom_fdim - atomic_num_block_w
        self.bond_fdim = 2 * single_bond_fdim

        self.atom_property_list_onehot = cuik_molmaker.atom_onehot_feature_names_to_array(
            atom_props_onehot
        )
        self.atom_property_list_float = cuik_molmaker.atom_float_feature_names_to_array(
            atom_props_float
        )
        self.bond_property_list = cuik_molmaker.bond_feature_names_to_array(bond_props)

        self._mode_int = int(cuik_molmaker.reaction_mode_to_int(self.reaction_mode.upper()))

    def __call__(
        self, reac_smiles_list: list[str], prod_smiles_list: list[str]
    ) -> BatchCuikMolGraph:
        (
            atom_feats,
            bond_feats,
            edge_index,
            rev_edge_index,
            batch,
        ) = cuik_molmaker.batch_reaction_featurizer(
            reac_smiles_list,
            prod_smiles_list,
            self.atom_property_list_onehot,
            self.atom_property_list_float,
            self.bond_property_list,
            self.keep_h,
            self.add_h,
            False,  # offset_carbon
            self._mode_int,
        )

        return BatchCuikMolGraph(
            V=torch.from_numpy(atom_feats),
            E=torch.from_numpy(bond_feats),
            edge_index=torch.from_numpy(edge_index),
            rev_edge_index=torch.from_numpy(rev_edge_index),
            batch=torch.from_numpy(batch),
        )
