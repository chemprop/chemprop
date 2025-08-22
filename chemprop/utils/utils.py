from __future__ import annotations

from enum import StrEnum
import os
from typing import Any, Callable, Iterable, Iterator, Type

import multiprocess
import numpy as np
import psutil
from rdkit import Chem


class EnumMapping(StrEnum):
    @classmethod
    def get(cls, name: str | EnumMapping) -> EnumMapping:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise KeyError(
                f"Unsupported {cls.__name__} member! got: '{name}'. expected one of: {', '.join(cls.keys())}"
            )

    @classmethod
    def keys(cls) -> Iterator[str]:
        return (e.name for e in cls)

    @classmethod
    def values(cls) -> Iterator[str]:
        return (e.value for e in cls)

    @classmethod
    def items(cls) -> Iterator[tuple[str, str]]:
        return zip(cls.keys(), cls.values())


def make_mol(
    smi: str,
    keep_h: bool = False,
    add_h: bool = False,
    ignore_stereo: bool = False,
    reorder_atoms: bool = False,
) -> Chem.Mol:
    """build an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool, optional
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps
        them if they are specified. Default is False.
    add_h : bool, optional
        whether to add hydrogens to the molecule. Default is False.
    ignore_stereo : bool, optional
        whether to ignore stereochemical information (R/S and Cis/Trans) when constructing the molecule. Default is False.
    reorder_atoms : bool, optional
        whether to reorder the atoms in the molecule by their atom map numbers. This is useful when
        the order of atoms in the SMILES string does not match the atom mapping, e.g. '[F:2][Cl:1]'.
        Default is False. NOTE: This does not reorder the bonds.

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h
    mol = Chem.MolFromSmiles(smi, params)

    if mol is None:
        raise RuntimeError(f"SMILES {smi} is invalid! (RDKit returned None)")

    if add_h:
        mol = Chem.AddHs(mol)

    if ignore_stereo:
        for atom in mol.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in mol.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)

    if reorder_atoms:
        atom_map_numbers = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
        new_order = np.argsort(atom_map_numbers).tolist()
        mol = Chem.rdmolops.RenumberAtoms(mol, new_order)

    return mol


def create_and_call_object(
    cls: Type,
    call_args: tuple = (),
    call_kwargs: dict = None,
    init_args: tuple = (),
    init_kwargs: dict = None,
) -> Any:
    """
    Instantiate a class with optional init args, then call the instance with args.
    This is useful for parallel calls to methods that contain boost functions.
    """
    if call_kwargs is None:
        call_kwargs = {}
    if init_kwargs is None:
        init_kwargs = {}

    return cls(*init_args, **init_kwargs)(*call_args, **call_kwargs)


def parallel_execute(
    exe_func: Callable,
    func_args: Iterable[tuple] = (),
    func_kwargs: Iterable[dict] = (),
    n_workers: int = 0,
) -> list:
    """Optionally executes a function in parallel.

    Parameters
    ----------
    exe_func : Callable
        function to execute.
    func_args : Iterable
        arguments for each iteration of function execution.
    func_kwargs : Iterable
        keyword arguments for each iteration of function execution.
    n_workers : int, optional
        Number of parallel workers.

    Returns
    -------
    list
        list of function outputs for each argument.
    """
    func_args = list(func_args)
    func_kwargs = list(func_kwargs)

    if not func_kwargs:
        func_kwargs = [{}] * len(func_args)
    if not func_args:
        func_args = [()] * len(func_kwargs)

    combined = list(zip(func_args, func_kwargs))

    if n_workers >= 2:

        def wrapped_call(args, kwargs):
            return exe_func(*args, **kwargs)

        with multiprocess.Pool(n_workers) as p:
            results = p.starmap(wrapped_call, combined)
    else:
        results = [exe_func(*func_arg, **func_kwargs) for (func_arg, func_kwargs) in combined]
    return results


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


def get_memory_usage():
    # Get the current process
    process = psutil.Process(os.getpid())

    # Get memory info in bytes
    memory_info = process.memory_info()

    # Convert to MB for readability
    memory_mb = memory_info.rss / 1024 / 1024

    return f"Memory usage: {memory_mb:.2f} MB"


def is_cuikmolmaker_available():
    try:
        import cuik_molmaker  # noqa: F401

        return True
    except ImportError:
        return False
