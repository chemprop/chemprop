import logging
import os
from os import PathLike
import sys
import warnings

import torch

from chemprop.models.serialization import (
    CHEMPROP_VERSION_KEY,
    WEIGHTS_ONLY_DEFAULT_WARNING,
    load_checkpoint,
    rebuild_saved_metrics,
    rebuild_saved_transforms,
    resolve_weights_only,
)
from chemprop.nn.hparams import serialize_hparams

logger = logging.getLogger(__name__)


def convert_model_dict_v2_1_to_v2_4(model_dict: dict) -> dict:
    hparams = model_dict["hyper_parameters"]
    rebuild_saved_metrics(hparams)
    rebuild_saved_transforms(hparams)

    model_dict = {
        key: value if key == "state_dict" else serialize_hparams(value)
        for key, value in (model_dict | {"hyper_parameters": hparams}).items()
    }
    model_dict.setdefault(CHEMPROP_VERSION_KEY, "2.4.0")

    return model_dict


def convert_model_file_v2_1_to_v2_4(
    model_file_in: PathLike, model_file_out: PathLike, *, weights_only: bool | None = None
) -> None:
    weights_only = resolve_weights_only(weights_only)

    model_dict = load_checkpoint(model_file_in, map_location="cpu", weights_only=weights_only)

    torch.save(convert_model_dict_v2_1_to_v2_4(model_dict), model_file_out)

    if hasattr(torch.serialization, "get_unsafe_globals_in_checkpoint"):
        unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(model_file_out)
        if unsafe_globals:
            logger.warning(
                f"The converted model file still contains arbitrary objects ({unsafe_globals}), so "
                "it can only be loaded with `weights_only=False`."
            )


if __name__ == "__main__":
    if os.getenv("CHEMPROP_ALLOW_UNSAFE_LOAD") != "1":
        warnings.warn(
            f"{WEIGHTS_ONLY_DEFAULT_WARNING} When running this script, set the environment "
            "variable `CHEMPROP_ALLOW_UNSAFE_LOAD=1` to silence this warning.",
            FutureWarning,
        )
    convert_model_file_v2_1_to_v2_4(sys.argv[1], sys.argv[2], weights_only=False)
