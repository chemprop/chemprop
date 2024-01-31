from os import PathLike
import torch
from sklearn.preprocessing import StandardScaler

from chemprop.models.model import MPNN
from chemprop.models.multi import MulticomponentMPNN

def save_model(path: PathLike, model: MPNN, input_scalers: list[StandardScaler] | None, output_scaler: StandardScaler | None) -> None:
    torch.save(
        {
            "hyper_parameters": model.hparams,
            "state_dict": model.state_dict(),
            "input_scalers": input_scalers,
            "output_scaler": output_scaler,
        },
        path,
    )


def load_model(path: PathLike) -> tuple[MPNN, list[StandardScaler] | None, StandardScaler | None]:
    try:
        model = MulticomponentMPNN.load_from_file(path)
    except KeyError: # MPNN
        model = MPNN.load_from_file(path)

    d = torch.load(path)
    input_scalers = d["input_scalers"]
    output_scaler = d["output_scaler"]

    return model, input_scalers, output_scaler
