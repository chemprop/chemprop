from os import PathLike

import torch

from chemprop.models.model import MPNN
from chemprop.models.mol_atom_bond import MolAtomBondMPNN
from chemprop.models.multi import MulticomponentMPNN


def save_model(
    path: PathLike,
    model: MPNN | MolAtomBondMPNN | MulticomponentMPNN,
    output_columns: list[str]
    | tuple[list[str] | None, list[str] | None, list[str] | None]
    | None = None,
) -> None:
    torch.save(
        {
            "hyper_parameters": model.hparams,
            "state_dict": model.state_dict(),
            "output_columns": output_columns,
        },
        path,
    )


def load_model(
    path: PathLike, multicomponent: bool = False, mol_atom_bond: bool = False
) -> MPNN | MulticomponentMPNN | MolAtomBondMPNN:
    model_cls = [
        [MPNN, MulticomponentMPNN],
        [MolAtomBondMPNN, "Atom/Bond predictions not supported for multicomponent"],
    ][mol_atom_bond][multicomponent]

    return model_cls.load_from_file(path, map_location=torch.device("cpu"))


def load_output_columns(
    path: PathLike,
) -> list[str] | tuple[list[str] | None, list[str] | None, list[str] | None] | None:
    model_file = torch.load(path, map_location=torch.device("cpu"), weights_only=False)

    return model_file.get("output_columns")
