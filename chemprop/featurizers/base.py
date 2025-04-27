from abc import abstractmethod
from collections.abc import Hashable, Sized
from typing import Callable, Generic, Sequence, TypeVar

import numpy as np

from chemprop.data.molgraph import MolGraph

S = TypeVar("S")
T = TypeVar("T")


class Featurizer(Generic[S, T]):
    """An :class:`Featurizer` featurizes inputs type ``S`` into outputs of
    type ``T``."""

    @abstractmethod
    def __call__(self, input: S, *args, **kwargs) -> T:
        """featurize an input"""


class VectorFeaturizer(Featurizer[S, np.ndarray], Sized):
    ...


class GraphFeaturizer(Featurizer[S, MolGraph]):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        ...


class Subfeature(Generic[S]):
    """Extract a one-hot encoding or a raw value from an Atom or Bond.

    This class uses a getter function to extract an attribute from an RDKit Atom or Bond.
    If a list of choices is provided, the attribute is converted into a one-hot encoded tuple.
    If choices are omitted, the raw attribute value is returned inside a tuple.

    Instances are callable: passing an Atom, Bond, or None returns the encoding.

    Parameters
    ----------
    getter : Callable[[Atom | Bond], Hashable]
        A function that extracts the attribute to be encoded from an Atom or Bond.
    choices : Sequence[Hashable], optional
        A sequence of possible values. If provided, the attribute is one-hot encoded.
        If None, the attribute value itself is used as the output.
    unknown_padding : bool, default=False
        If True and choices are given, adds an extra dimension to handle unknown values.

    Raises
    ------
    ValueError
        If `choices` are not unique.

    Example
    -------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C(O)N")
    >>> symbol_subfeature = Subfeature[Chem.Atom](
    ...     getter=lambda atom: atom.GetSymbol(),
    ...     choices=["C", "N", "O"],
    ...     unknown_padding=True,
    ... )
    >>> atom = mol.GetAtomWithIdx(0)
    >>> symbol_subfeature(atom)
    (1, 0, 0, 0)
    >>> symbol_subfeature.as_string(atom)
    '1000'

    >>> mass_subfeature = Subfeature[Chem.Atom](getter=lambda atom: atom.GetMass())
    >>> mass_subfeature(atom)
    (12.011,)
    >>> mass_subfeature.as_string(atom)
    '12.011'

    >>> bond = mol.GetBondWithIdx(0)
    >>> bond_type_subfeature = Subfeature[Chem.Atom](
    ...     getter=lambda bond: bond.GetBondType(),
    ...     choices=[Chem.BondType.SINGLE, Chem.BondType.DOUBLE],
    ... )
    >>> bond_type_subfeature(bond)
    (1, 0)
    >>> bond_type_subfeature.as_string(bond)
    '10'

    """

    def __init__(
        self,
        getter: Callable[[S], Hashable],
        choices: Sequence[Hashable] | None = None,
        unknown_padding: bool = False,
    ):
        """Initialize the Subfeature."""
        self.getter = getter
        self.choices = choices
        self.unknown_padding = unknown_padding
        if self.choices is not None and len(self.choices) != len(set(self.choices)):
            raise ValueError("choices must be unique")

    def __len__(self) -> int:
        """Return the length of the feature vector."""
        if self.choices is None:
            return 1
        return len(self.choices) + int(self.unknown_padding)

    def __call__(self, entity: S | None) -> tuple[int | float, ...]:
        """Encode an Atom or Bond as a one-hot tuple or a raw value tuple.

        Parameters
        ----------
        entity : Atom | Bond | None
            The input entity. If None, returns a tuple of zeros.

        Returns
        -------
        tuple[int | float, ...]
            One-hot encoded tuple if choices are provided;
            otherwise, a tuple with the raw extracted value.
        """
        if entity is None:
            return (0,) * len(self)
        if self.choices is None:
            return (self.getter(entity),)
        lst = [x == self.getter(entity) for x in self.choices]
        if self.unknown_padding:
            lst.append(not any(lst))
        return tuple(map(int, lst))

    def as_string(self, entity: S | None, decimals: int = 3) -> str:
        """Return a string representation of the feature encoding.

        Parameters
        ----------
        entity : Atom | Bond | None
            The input entity. If None, returns a string of zeros.
        decimals : int, default=3
            Number of decimals if the value is a float (only relevant if choices are None).

        Returns
        -------
        str
            The string encoding of the feature vector.
        """
        if self.choices is None:
            x = self(entity)[0]
            if isinstance(x, float):
                return f"{x:.{decimals}f}"
            return str(int(x))
        return "".join(map(str, self(entity)))


class MultiHotFeaturizer(VectorFeaturizer[S]):
    """A vector featurizer that concatenates multiple subfeatures.

    Parameters
    ----------
    *subfeats : Subfeature
        The subfeatures to concatenate.

    Example
    -------
    >>> from rdkit import Chem
    >>> symbol_subfeature = Subfeature[Chem.Atom](
    ...     getter=lambda atom: atom.GetSymbol(),
    ...     choices=["C", "N", "O"],
    ...     unknown_padding=True,
    ... )
    >>> mass_subfeature = Subfeature[Chem.Atom](
    ...     getter=lambda atom: 0.01 * atom.GetMass()
    ... )
    >>> featurizer = MultiHotFeaturizer[Chem.Atom](
    ...     symbol_subfeature, mass_subfeature
    ... )
    >>> mol = Chem.MolFromSmiles("C(O)N")
    >>> atom = mol.GetAtomWithIdx(0)
    >>> featurizer.to_string(atom)
    '1000 0.120'

    """

    def __init__(self, *subfeats: Subfeature):
        self._subfeats = subfeats
        self._subfeat_sizes = list(map(len, subfeats))
        self._size = sum(self._subfeat_sizes)

    def __len__(self):
        return self._size

    def __call__(self, input: S) -> np.ndarray:
        return np.concatenate([f(input) for f in self._subfeats])

    def to_string(self, input: S) -> str:
        """Return a string representation of the concatenated subfeatures.

        Parameters
        ----------
        input : Atom | Bond
            The input entity.

        Returns
        -------
        str
            The string encoding of the concatenated subfeatures, with spaces separating each subfeature.

        """
        return " ".join(f.as_string(input) for f in self._subfeats)
