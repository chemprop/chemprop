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


if __name__ == "__main__":
    model = torch.load(
        sys.argv[1], map_location="cpu", pickle_module=sys.modules[__name__], weights_only=False
    )
    torch.save(model, sys.argv[2])
