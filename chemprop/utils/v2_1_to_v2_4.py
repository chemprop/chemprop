import logging
from os import PathLike
import sys

import torch

from chemprop.models.serialization import rebuild_saved_metrics, rebuild_saved_transforms
from chemprop.nn.hparams import serialize_hparams
from chemprop.utils.v1_to_v2 import convert_model_dict_v1_to_v2
from chemprop.utils.v2_0_to_v2_1 import load_model_file_v2_0

logger = logging.getLogger(__name__)


def convert_model_dict_v2_1_to_v2_4(model_dict: dict) -> dict:
    """Converts a v2.1+ model dictionary, which may contain arbitrary objects, to a dictionary
    that can be saved and then loaded with ``weights_only=True``"""
    hparams = dict(model_dict["hyper_parameters"])
    rebuild_saved_metrics(hparams)
    rebuild_saved_transforms(hparams)

    # the other keys of lightning checkpoints (e.g., the optimizer states) may contain numpy objects
    return {
        key: value if key == "state_dict" else serialize_hparams(value)
        for key, value in (model_dict | {"hyper_parameters": hparams}).items()
    }


def _save_model_dict(model_dict: dict, model_file: PathLike) -> None:
    torch.save(model_dict, model_file)

    if hasattr(torch.serialization, "get_unsafe_globals_in_checkpoint"):
        unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(model_file)
        if unsafe_globals:
            logger.warning(
                f"The converted model file still contains arbitrary objects ({unsafe_globals}), so "
                "it can only be loaded with `weights_only=False`."
            )


def convert_model_file_v2_1_to_v2_4(model_file_in: PathLike, model_file_out: PathLike) -> None:
    """Converts a v2.1+ model file, which may contain arbitrary objects, to a model file that can
    be loaded with ``weights_only=True``.

    .. warning::
        The input file is loaded with ``weights_only=False``, which can execute arbitrary code, so
        only convert model files you trust.
    """
    model_dict = torch.load(model_file_in, map_location="cpu", weights_only=False)
    _save_model_dict(convert_model_dict_v2_1_to_v2_4(model_dict), model_file_out)


def convert_model_file_to_latest(model_file_in: PathLike, model_file_out: PathLike) -> None:
    """Converts a model file from any version of Chemprop to the latest format, which can be
    loaded with ``weights_only=True``.

    .. warning::
        The input file is loaded with ``weights_only=False``, which can execute arbitrary code, so
        only convert model files you trust.
    """
    model_dict = load_model_file_v2_0(model_file_in)

    if "hyper_parameters" not in model_dict and "args" in model_dict:
        logger.info("Converting v1 model to v2...")
        model_dict = convert_model_dict_v1_to_v2(model_dict)
        logger.warning(
            "Remember to use the same featurizers which were used when training the model. The "
            "default v1 atom featurizer is `chemprop.featurizers.atom.MultiHotAtomFeaturizer.v1()` "
            "and can be specified from the command line with `--multi-hot-atom-featurizer-mode v1`."
        )

    _save_model_dict(convert_model_dict_v2_1_to_v2_4(model_dict), model_file_out)


if __name__ == "__main__":
    convert_model_file_to_latest(sys.argv[1], sys.argv[2])
