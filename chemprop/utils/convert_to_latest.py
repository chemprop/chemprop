import logging
from os import PathLike

import torch

from chemprop.models.serialization import resolve_weights_only
from chemprop.utils import v2_0_to_v2_1
from chemprop.utils.v1_to_v2 import convert_model_dict_v1_to_v2
from chemprop.utils.v2_1_to_v2_4 import convert_model_dict_v2_1_to_v2_4

logger = logging.getLogger(__name__)

CONVERSIONS = [convert_model_dict_v1_to_v2, convert_model_dict_v2_1_to_v2_4]


def convert_model_file_to_latest(
    model_file_in: PathLike, model_file_out: PathLike, *, weights_only: bool | None = None
) -> None:
    model_dict = torch.load(
        model_file_in,
        map_location="cpu",
        pickle_module=v2_0_to_v2_1,
        weights_only=resolve_weights_only(weights_only),
    )

    errors = {}
    for convert in CONVERSIONS:
        try:
            model_dict = convert(model_dict)
        except Exception as e:
            logger.debug(f"Skipping `{convert.__name__}`: {e!r}")
            errors[convert.__name__] = e
            continue

        logger.debug(f"Applied `{convert.__name__}`")
        if convert is convert_model_dict_v1_to_v2:
            logger.warning(
                "Remember to use the same featurizers which were used when training the model. "
                "The default v1 atom featurizer is "
                "`chemprop.featurizers.atom.MultiHotAtomFeaturizer.v1()` and can be specified "
                "from the command line with `--multi-hot-atom-featurizer-mode v1`."
            )

    if len(errors) == len(CONVERSIONS):
        raise RuntimeError(f"Could not apply any conversion to '{model_file_in}'. Errors: {errors}")

    torch.save(model_dict, model_file_out)

    if hasattr(torch.serialization, "get_unsafe_globals_in_checkpoint"):
        unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(model_file_out)
        if unsafe_globals:
            logger.warning(
                f"The converted model file still contains arbitrary objects ({unsafe_globals}), so "
                "it can only be loaded with `weights_only=False`."
            )
