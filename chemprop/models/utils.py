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
            "input_scalers": model.input_transform.input_scalers,
            "output_scaler": model.output_transform.output_scaler
        },
        path,
    )


def load_model(path: PathLike, multicomponent: bool) -> tuple[MPNN, dict | None, StandardScaler | None]:
    if multicomponent:
        model = MulticomponentMPNN.load_from_file(path)
    else:
        model = MPNN.load_from_file(path)

    d = torch.load(path)
    input_scalers = d.get("input_scalers", None)
    output_scaler = d.get("output_scaler", None)
    model.input_scalers = InputTransform(input_scalers)
    model.output_transform = OutputTransform(output_scaler)

    return model

