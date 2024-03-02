from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from collections.abc import Callable

import numpy as np
from rdkit import Chem

from chemprop.featurizers.molgraph.molgraph import MolGraph

T = TypeVar("T", Chem.Mol, tuple[Chem.Mol, Chem.Mol])


class MolGraphFeaturizer(Generic[T], Callable):
    """A :class:`MolGraphFeaturizer` featurizes input molecules or reactions
    into :class:`MolGraph`\s"""

    @abstractmethod
    def __call__(
        self,
        input: T,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        """Featurize the input into a molecular graph

        Parameters
        ----------
        mol : T
            the input molecule or reaction
        atom_features_extra : np.ndarray | None, default=None
            Additional features to concatenate to the calculated atom features
        bond_features_extra : np.ndarray | None, default=None
            Additional features to concatenate to the calculated bond features

        Returns
        -------
        MolGraph
            the corresponding molecular graph
        """
