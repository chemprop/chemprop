import os
from os import PathLike
import pickle
import sys
import warnings

import torch

from chemprop.models.serialization import WEIGHTS_ONLY_DEFAULT_WARNING, resolve_weights_only


class Unpickler(pickle.Unpickler):
    name_mappings = {
        "MSELoss": "MSE",
        "MSEMetric": "MSE",
        "MAEMetric": "MAE",
        "RMSEMetric": "RMSE",
        "BoundedMSELoss": "BoundedMSE",
        "BoundedMSEMetric": "BoundedMSE",
        "BoundedMAEMetric": "BoundedMAE",
        "BoundedRMSEMetric": "BoundedRMSE",
        "SIDLoss": "SID",
        "SIDMetric": "SID",
        "WassersteinLoss": "Wasserstein",
        "WassersteinMetric": "Wasserstein",
        "R2Metric": "R2Score",
        "BinaryAUROCMetric": "BinaryAUROC",
        "BinaryAUPRCMetric": "BinaryAUPRC",
        "BinaryAccuracyMetric": "BinaryAccuracy",
        "BinaryF1Metric": "BinaryF1Score",
        "BCEMetric": "BCELoss",
    }

    def find_class(self, module, name):
        if module == "chemprop.nn.loss":
            module = "chemprop.nn.metrics"
        name = self.name_mappings.get(name, name)
        return super().find_class(module, name)


def convert_model_file_v2_0_to_v2_1(
    model_v1_file: PathLike, model_v2_file: PathLike, *, weights_only: bool | None = None
):
    model = torch.load(
        model_v1_file,
        map_location="cpu",
        pickle_module=sys.modules[__name__],
        weights_only=resolve_weights_only(weights_only),
    )
    torch.save(model, model_v2_file)


if __name__ == "__main__":
    if os.getenv("CHEMPROP_ALLOW_UNSAFE_LOAD") != "1":
        warnings.warn(
            f"{WEIGHTS_ONLY_DEFAULT_WARNING} When running this script, set the environment "
            "variable `CHEMPROP_ALLOW_UNSAFE_LOAD=1` to silence this warning.",
            FutureWarning,
        )
    convert_model_file_v2_0_to_v2_1(sys.argv[1], sys.argv[2], weights_only=False)
