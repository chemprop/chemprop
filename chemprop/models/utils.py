from os import PathLike
import torch
from sklearn.preprocessing import StandardScaler

from chemprop.models.model import MPNN, OutputTransform, InputTransform
from chemprop.models.multi import MulticomponentMPNN

def save_model(path: PathLike, model: MPNN) -> None:
    torch.save(
        {
            "hyper_parameters": model.hparams,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(path: PathLike, multicomponent: bool) -> tuple[MPNN, dict | None, StandardScaler | None]:
    if multicomponent:
        model = MulticomponentMPNN.load_from_file(path)
    else:
        model = MPNN.load_from_file(path)

    return model

