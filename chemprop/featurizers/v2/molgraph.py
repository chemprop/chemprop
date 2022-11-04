from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


class MolGraph(NamedTuple):
    """A `MolGraph` represents the graph structure and featurization of a single molecule.

    Attributes
    ----------
    n_atoms : int
        the number of atoms in the molecule
    n_bonds : int
        the number of bonds in the molecule
    X_v : np.ndarray
        the atom features of the molecule
    X_e : np.ndarray
        the bond features of the molecule
    a2b : list[tuple[int]]
        A mapping from an atom index to a list of incoming bond indices.
    b2a : np.ndarray
        A mapping from a bond index to the index of the atom the bond originates from.
    b2revb : np.ndarray
        A mapping from a bond index to the index of the reverse bond.
    a2a : Optional[list[int]]
    b2b: Optional[np.ndarray]
    """

    n_atoms: int
    n_bonds: int
    X_v: np.ndarray
    X_e: np.ndarray
    a2b: list[tuple[int]]
    b2a: list[int]
    b2revb: np.ndarray
    a2a: Optional[list[int]]
    b2b: Optional[np.ndarray]
