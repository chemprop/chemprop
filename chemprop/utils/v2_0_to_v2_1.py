from os import PathLike
import pickle
import sys

import torch


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


def load_model_file_v2_0(model_file: PathLike) -> dict:
    """Load a model file with the classes renamed in v2.1 mapped to their new names. Files from
    other versions are loaded unchanged.

    .. warning::
        The file is loaded with ``weights_only=False``, which can execute arbitrary code, so only
        load model files you trust.
    """
    return torch.load(
        model_file, map_location="cpu", pickle_module=sys.modules[__name__], weights_only=False
    )


def convert_model_file_v2_0_to_v2_1(model_v1_file: PathLike, model_v2_file: PathLike):
    model = load_model_file_v2_0(model_v1_file)
    torch.save(model, model_v2_file)


if __name__ == "__main__":
    convert_model_file_v2_0_to_v2_1(sys.argv[1], sys.argv[2])
