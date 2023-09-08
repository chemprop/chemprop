from __future__ import annotations

import csv
from enum import Enum
import os
from typing import List, Union, Callable, Any, Iterable

from rdkit import Chem
from functools import wraps
from time import time
from datetime import timedelta
import logging 

# copied over from v1
def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:
    """
    Creates a decorator which wraps a function with a timer that prints the elapsed time.

    :param logger_name: The name of the logger used to record output. If None, uses :code:`print` instead.
    :return: A decorator which wraps a function with a timer that prints the elapsed time.
    """

    def timeit_decorator(func: Callable) -> Callable:
        """
        A decorator which wraps a function with a timer that prints the elapsed time.

        :param func: The function to wrap with the timer.
        :return: The function wrapped with the timer.
        """

        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time() - start_time))
            info = logging.getLogger(logger_name).info if logger_name is not None else print
            info(f"Elapsed time = {delta}")

            return result

        return wrap

    return timeit_decorator

class AutoName(Enum):
    # todo: figure out where this function is supposed to be used
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    def __str__(self) -> str:
        return self.value

    @classmethod
    def get(cls, name: str | AutoName) -> AutoName:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(
                f"Unsupported {cls.__name__} alias! got: '{name}'. expected one of: {cls.keys()}"
            )

    @classmethod
    def keys(cls) -> set[str]:
        return {e.value for e in cls}


def make_mol(smi: str, keep_h: bool, add_h: bool) -> Chem.Mol:
    """build an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified
    add_h : bool
        whether to add hydrogens to the molecule

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        )
    else:
        mol = Chem.MolFromSmiles(smi)

    return Chem.AddHs(mol) if add_h else mol


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def preprocess_smiles_columns(path: str,
                              smiles_columns: Union[str, List[str]] = None,
                              number_of_molecules: int = 1) -> List[str]:
    """
    Preprocesses the :code:`smiles_columns` variable to ensure that it is a list of column
    headings corresponding to the columns in the data file holding SMILES. Assumes file has a header.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules with associated SMILES for each
                           data point.
    :return: The preprocessed version of :code:`smiles_columns` which is guaranteed to be a list.
    """

    if smiles_columns is None:
        if os.path.isfile(path):
            columns = get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None]*number_of_molecules
    else:
        if not isinstance(smiles_columns,list):
            smiles_columns=[smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError('Length of smiles_columns must match number_of_molecules.')
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError('Provided smiles_columns do not match the header of data file.')

    return smiles_columns

def pretty_shape(shape: Iterable[int]) -> str:
    """Make a pretty string from an input shape

    Example
    --------
    >>> X = np.random.rand(10, 4)
    >>> X.shape
    (10, 4)
    >>> pretty_shape(X.shape)
    '10 x 4'
    """
    return " x ".join(map(str, shape))
