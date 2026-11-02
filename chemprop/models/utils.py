from os import PathLike

import torch

import chemprop
from chemprop.models.model import MPNN
from chemprop.models.mol_atom_bond import MolAtomBondMPNN
from chemprop.models.multi import MulticomponentMPNN
from chemprop.models.serialization import CHEMPROP_VERSION_KEY, load_checkpoint
from chemprop.nn.hparams import serialize_hparams


def save_model(
    path: PathLike,
    model: MPNN | MolAtomBondMPNN | MulticomponentMPNN,
    output_columns: list[str]
    | tuple[list[str] | None, list[str] | None, list[str] | None]
    | None = None,
) -> None:
    torch.save(
        {
            "hyper_parameters": serialize_hparams(model.hparams),
            "state_dict": model.state_dict(),
            "output_columns": output_columns,
            CHEMPROP_VERSION_KEY: chemprop.__version__,
        },
        path,
    )


def load_model(
    path: PathLike,
    multicomponent: bool = False,
    mol_atom_bond: bool = False,
    weights_only: bool | None = None,
) -> MPNN | MulticomponentMPNN | MolAtomBondMPNN:
    model_cls = [
        [MPNN, MulticomponentMPNN],
        [MolAtomBondMPNN, "Atom/Bond predictions not supported for multicomponent"],
    ][mol_atom_bond][multicomponent]

    return model_cls.load_from_file(
        path, map_location=torch.device("cpu"), weights_only=weights_only
    )


def load_output_columns(
    path: PathLike, weights_only: bool | None = None
) -> list[str] | tuple[list[str] | None, list[str] | None, list[str] | None] | None:
    model_file = load_checkpoint(path, map_location=torch.device("cpu"), weights_only=weights_only)

    return model_file.get("output_columns")
