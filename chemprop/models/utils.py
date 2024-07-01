from os import PathLike

import torch

from chemprop.models.model import MPNN
from chemprop.models.multi import MulticomponentMPNN


def save_model(path: PathLike, model: MPNN) -> None:
    torch.save({"hyper_parameters": model.hparams, "state_dict": model.state_dict()}, path)


def load_model(path: PathLike, multicomponent: bool, accelerator: str) -> MPNN:
    if accelerator == "cpu":
        map_location = torch.device("cpu")
    else:
        map_location = None

    if multicomponent:
        model = MulticomponentMPNN.load_from_file(path, map_location=map_location)
    else:
        model = MPNN.load_from_file(path, map_location=map_location)

    return model
