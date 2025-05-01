from abc import abstractmethod
from collections.abc import Hashable, Sized
from typing import Any, Callable, Generic, Sequence, TypeVar

import numpy as np

from chemprop.data.molgraph import MolGraph

S = TypeVar("S")
T = TypeVar("T")


class Featurizer(Generic[S, T]):
    """A :class:`Featurizer` featurizes inputs of type ``S`` into outputs of type ``T``."""

    @abstractmethod
    def __call__(self, input: S | None, *args, **kwargs) -> T:
        """Featurize an input."""


class VectorFeaturizer(Featurizer[S, np.ndarray], Sized):
    ...


class GraphFeaturizer(Featurizer[S, MolGraph]):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        ...


class OneHotFeaturizer(VectorFeaturizer[S]):
    """Extract a one-hot encoding from an input object.

    This class uses a getter function to extract an attribute from an input object and converts
    the attribute into a one-hot encoded vector.

    Parameters
    ----------
    getter : Callable[[S], Hashable]
        A function that extracts the attribute to be encoded from an input object.
    choices : Sequence[Hashable]
        A sequence of unique possible values.
    padding : bool, default=False
        If True, adds an extra dimension to handle unknown values.

    Raises
    ------
    ValueError
        If the provided `choices` are not unique.

    Example
    -------
    >>> from rdkit import Chem
    >>> symbol_featurizer = OneHotFeaturizer[Chem.Atom](
    ...     getter=lambda atom: atom.GetSymbol(),
    ...     choices=["C", "N", "O"],
    ...     padding=True,
    ... )
    >>> mol = Chem.MolFromSmiles("C(O)N")
    >>> atom = mol.GetAtomWithIdx(0)
    >>> symbol_featurizer(atom)
    array([1, 0, 0, 0])
    >>> symbol_featurizer.to_string(atom)
    '1000'

    >>> bond_type_featurizer = OneHotFeaturizer[Chem.Atom](
    ...     getter=lambda bond: bond.GetBondType(),
    ...     choices=[Chem.BondType.SINGLE, Chem.BondType.DOUBLE],
    ... )
    >>> bond = mol.GetBondWithIdx(0)
    >>> bond_type_featurizer(bond)
    array([1, 0])
    >>> bond_type_featurizer.to_string(bond)
    '10'

    """

    def __init__(
        self, getter: Callable[[S], Hashable], choices: Sequence[Hashable], padding: bool = False
    ):
        self.getter = getter
        self.choices = choices
        self.padding = padding
        if len(self.choices) != len(set(self.choices)):
            raise ValueError("choices must be unique")

    def __len__(self) -> int:
        """Return the length of the feature vector."""
        return len(self.choices) + int(self.padding)

    def __call__(self, input: S | None) -> np.ndarray:
        """Encode an input object as a one-hot vector.

        Parameters
        ----------
        input : S | None
            The input object. If None, returns a vector of zeros.

        Returns
        -------
        np.ndarray
            One-hot encoded vector.

        """
        vector = np.zeros(len(self), dtype=int)
        if input is None:
            return vector
        option = self.getter(input)
        try:
            vector[self.choices.index(option)] = 1
        except ValueError:
            if self.padding:
                vector[-1] = 1
        return vector

    def to_string(self, input: S | None, _: int = 3) -> str:
        """Return a string representation of the feature encoding.

        Parameters
        ----------
        input : S | None
            The input entity. If None, returns a string of zeros.

        Returns
        -------
        str
            The string encoding of the feature vector.
        """
        if input is None:
            return "0" * len(self)
        return "".join(map(str, self(input)))


class ValueFeaturizer(VectorFeaturizer[S]):
    """Extract a raw value from an input object.

    This class uses a getter function to extract an attribute from an input object and converts
    the attribute into a single-element vector.

    Parameters
    ----------
    getter : Callable[[S], Hashable]
        A function that extracts the attribute to be encoded from an input object.
    dtype : type
        The data type of the output vector.

    Example
    -------
    >>> from rdkit import Chem
    >>> mass_featurizer = ValueFeaturizer[Chem.Atom](
    ...     getter=lambda atom: atom.GetMass(), dtype=float
    ... )
    >>> mol = Chem.MolFromSmiles("C(O)N")
    >>> atom = mol.GetAtomWithIdx(0)
    >>> mass_featurizer(atom)
    array([12.011])
    >>> mass_featurizer.to_string(atom)
    '12.011'

    """

    def __init__(self, getter: Callable[[S], Hashable], dtype: type):
        self.getter = getter
        self.dtype = dtype

    def __len__(self) -> int:
        """Return the length of the feature vector."""
        return 1

    def __call__(self, input: S | None) -> np.ndarray:
        """Encode a raw value as a vector.

        Parameters
        ----------
        input : S | None
            The input object. If None, returns a vector with a zero value.

        Returns
        -------
        np.ndarray
            A vector with the raw extracted value.

        """
        if input is None:
            return np.zeros(1, dtype=self.dtype)
        return np.array([self.getter(input)], dtype=self.dtype)

    def to_string(self, input: S | None, decimals: int = 3) -> str:
        """Return a string representation of the feature encoding.

        Parameters
        ----------
        input : S | None
            The input entity. If None, returns '0'.
        decimals : int, default=3
            Number of decimals (only relevant if the feature is a float value).

        Returns
        -------
        str
            The string encoding of the feature vector.
        """
        if input is None:
            return "0"
        x = self(input).item()
        if isinstance(x, float):
            return f"{x:.{decimals}f}"
        return str(int(x))


class NullityFeaturizer(VectorFeaturizer[Any]):
    """A subfeaturizer that encodes whether an input is None."""

    def __len__(self) -> int:
        return 1

    def __call__(self, input: Any) -> np.ndarray:
        return np.array([int(input is None)], dtype=int)

    def to_string(self, input: Any, _: int = 3) -> str:
        return "1" if input is None else "0"


class MultiHotFeaturizer(VectorFeaturizer[S]):
    """A vector featurizer that concatenates multiple subfeaturizers.

    Parameters
    ----------
    *subfeats : Subfeaturizer
        The subfeatures to concatenate.

    Example
    -------
    >>> from rdkit import Chem
    >>> symbol_featurizer = OneHotFeaturizer[Chem.Atom](
    ...     getter=lambda atom: atom.GetSymbol(),
    ...     choices=["C", "N", "O"],
    ...     padding=True,
    ... )
    >>> mass_featurizer = ValueFeaturizer[Chem.Atom](
    ...     getter=lambda atom: 0.01 * atom.GetMass(), dtype=float
    ... )
    >>> featurizer = MultiHotFeaturizer[Chem.Atom](
    ...     NullityFeaturizer(), symbol_featurizer, mass_featurizer
    ... )
    >>> mol = Chem.MolFromSmiles("C(O)N")
    >>> atom = mol.GetAtomWithIdx(0)
    >>> featurizer.to_string(atom)
    '0 1000 0.120'
    >>> featurizer.to_string(None)
    '1 0000 0'
    """

    def __init__(self, *subfeats: OneHotFeaturizer[S] | ValueFeaturizer[S] | NullityFeaturizer):
        self.subfeats = subfeats
        self._subfeat_sizes = list(map(len, subfeats))
        self._size = sum(self._subfeat_sizes)

    def __len__(self):
        return self._size

    def __call__(self, input: S | None) -> np.ndarray:
        return np.concatenate([f(input) for f in self.subfeats])

    def to_string(self, input: S | None, decimals: int = 3) -> str:
        """Return a string representation of the concatenated subfeatures.

        Parameters
        ----------
        input : S | None
            The input object.

        Returns
        -------
        str
            The string encoding of the concatenated subfeatures, with spaces separating each
            subfeature.

        """
        return " ".join(f.to_string(input, decimals) for f in self.subfeats)
