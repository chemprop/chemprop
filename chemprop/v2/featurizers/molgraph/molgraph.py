from typing import NamedTuple

import numpy as np


class MolGraph(NamedTuple):
    """A :class:`MolGraph` represents the graph featurization of a molecule."""

    V: np.ndarray
    """an array of shape ``V x d_v`` containing the atom features of the molecule"""
    E: np.ndarray
    """an array of shape ``E x d_e`` containing the bond features of the molecule"""
    edge_index: np.ndarray
    """an array of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: np.ndarray
    """A array of shape ``E`` that maps from an edge index to the index of the source of the reverse edge in :attr:`edge_index` attribute."""
