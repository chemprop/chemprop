from os import PathLike

import torch

from chemprop.models.model import MPNN
from chemprop.models.multi import MulticomponentMPNN


def save_model(path: PathLike, model: MPNN, output_columns: list[str] = None) -> None:
    torch.save(
        {
            "hyper_parameters": model.hparams,
            "state_dict": model.state_dict(),
            "output_columns": output_columns,
        },
        path,
    )


def load_model(path: PathLike, multicomponent: bool) -> MPNN:
    if multicomponent:
        model = MulticomponentMPNN.load_from_file(path, map_location=torch.device("cpu"))
    else:
        model = MPNN.load_from_file(path, map_location=torch.device("cpu"))

    return model


def load_output_columns(path: PathLike) -> list[str] | None:
    model_file = torch.load(path, map_location=torch.device("cpu"))

    return model_file.get("output_columns")
