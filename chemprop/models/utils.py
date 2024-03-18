from os import PathLike
import torch
from sklearn.preprocessing import StandardScaler

from chemprop.models.model import MPNN
from chemprop.models.multi import MulticomponentMPNN


def save_model(
    path: PathLike,
    model: MPNN,
    input_scalers: dict[str, StandardScaler] | None,
    output_scaler: StandardScaler | None,
) -> None:
    torch.save(
        {
            "hyper_parameters": model.hparams,
            "state_dict": model.state_dict(),
            "input_scalers": input_scalers,
            "output_scaler": output_scaler,
        },
        path,
    )


def load_model(
    path: PathLike, multicomponent: bool
) -> tuple[MPNN, dict[str, StandardScaler] | None, StandardScaler | None]:
    if multicomponent:
        model = MulticomponentMPNN.load_from_file(path)
    else:
        model = MPNN.load_from_file(path)

    d = torch.load(path)
    input_scalers = d.get("input_scalers", None)
    output_scaler = d.get("output_scaler", None)

    return model, input_scalers, output_scaler
