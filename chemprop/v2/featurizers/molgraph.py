from typing import NamedTuple

import numpy as np


class MolGraph(NamedTuple):
    """A :class:`MolGraph` represents the graph featurization of a molecule."""

    n_atoms: int
    """the number of atoms in the molecule"""
    n_bonds: int
    """the number of bonds in the molecule"""
    V: np.ndarray
    """an array of shape `V x d_v` containing the atom features of the molecule"""
    E: np.ndarray
    """an array of shape `E x d_e` containing the bond features of the molecule"""
    a2b: list[tuple[int]]
    """A list of length `V` that maps from an atom index to a list of incoming bond indices."""
    b2a: list[int]
    """A list of length `E` that maps from a bond index to the index of the atom the bond
    originates from."""
    b2revb: np.ndarray
    """A list of length `E` that maps from a bond index to the index of the reverse bond."""
    a2a: list[int] | None
    """a mapping from atom index to the indices of connected atoms"""
    b2b: np.ndarray | None
    """a mapping from bond index to the indices of connected bonds"""
