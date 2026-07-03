import json
import logging
from pathlib import Path
import typing

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from myerson.chemprop_explain import (
    MyersonClassExplainer,
    MyersonExplainer,
    MyersonSamplingClassExplainer,
    MyersonSamplingExplainer,
)
from myerson.chemprop_explain.utils import unbatch
import numpy as np
import torch

from chemprop.callbacks import CallbackRegistry
from chemprop.cli.common import find_models

logger = logging.getLogger(__name__)


@CallbackRegistry.register("myerson")
class MyersonExplainerCallback(Callback):
    """A :class:`MyersonExplainerCallback` calculates and saves Myerson explanations during a `predict` call.

    The explanations are saved as a compressed NumPy archive (:code:`.npz` file) by default.
    Each molecule's explanation is saved as a separate array within the archive (e.g., :code:`arr_0`, :code:`arr_1`, etc.).
    Each array will be a 1D or 2D NumPy array of shape :code:`num_atoms` (for regression or binary classification)
    or :code:`num_atoms x num_classes` (for multi-label classification) containing the explanation for one molecule.

    Alternatively, if :code:`save_as_json` is set to `True`, the explanations are saved as a JSON file.
    The JSON file contains a list of explanations, where each explanation corresponds to a molecule. For 2D explanations (multi-class), each inner list represents a column (i.e., attributions for a specific class across all atoms).

    Parameters
    ----------
    model_paths : list[Path]
        A list of paths to the models to be used for explanations.
    output : Path
        The path to the output file for saving predictions, used to derive the explanation file path.
    sampling_threshold : int, default=20
        The maximum number of atoms in a molecule for which to use the exact explainer. For
        molecules with more atoms, a sampling-based explainer is used.
    save_as_json : bool, default=False
        If `True`, save the explanations as a JSON file instead of a npz file.
    """

    def __init__(
        self,
        model_paths: list[Path],
        output: Path,
        sampling_threshold: int = 20,
        save_as_json: bool = False,
    ):
        super().__init__()
        self.sampling_threshold = sampling_threshold
        self.save_as_json = save_as_json

        logger.warning(
            "The 'myerson' callback can be computationally expensive and may significantly increase "
            "runtime, especially with large batch sizes."
        )

        model_paths = find_models(model_paths)

        model_file = torch.load(
            model_paths[0], map_location=torch.device("cpu"), weights_only=False
        )
        mol_atom_bond = "atom_predictor" in model_file["hyper_parameters"].keys()

        if mol_atom_bond:
            raise NotImplementedError(
                "Myerson Explanations are not supported for atom/bond level predictions."
            )
        if len(model_paths) > 1:
            logger.warning(
                f"Calculating Myerson explanations for multiple models ({len(model_paths)}) might take a long time."
            )

        logging.getLogger("MyersonExplainer").setLevel(logging.ERROR)
        logging.getLogger("MyersonSamplingExplainer").setLevel(logging.ERROR)
        logging.getLogger("MyersonClassExplainer").setLevel(logging.ERROR)
        logging.getLogger("MyersonSamplingClassExplainer").setLevel(logging.ERROR)

        self.model_counter = 0
        self.max_model_counter = len(model_paths) - 1

        self.output_filename_base = output.stem + "_myerson_explanation"
        self.output_path_dir = output.parent

    def on_predict_start(self, trainer, pl_module):
        if pl_module.predictor.__class__.__name__ not in [
            "BinaryClassificationFFN",
            "RegressionFFN",
        ]:
            raise NotImplementedError(
                f"Myerson explanations are only implemented for BinaryClassificationFNN and RegressionFFN. Got {pl_module.predictor.__class__.__name__}"
            )
        self.mol_idxs = []
        self.explanations = []

    @property
    def _last_mol_id(self) -> int:
        if len(self.mol_idxs) == 0:
            return -1
        return self.mol_idxs[-1]

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: typing.Any,
        batch: typing.Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        explainer_cls = MyersonExplainer if outputs.shape[1] == 1 else MyersonClassExplainer
        sampler_cls = (
            MyersonSamplingExplainer if outputs.shape[1] == 1 else MyersonSamplingClassExplainer
        )

        molgraphs = unbatch(batch.bmg)

        with torch.no_grad():
            for i, mg in enumerate(molgraphs, start=self._last_mol_id + 1):
                num_nodes = mg.V.shape[0]
                if num_nodes > self.sampling_threshold:
                    sampler = sampler_cls(mg, pl_module)
                    my_values = sampler.sample_all_myerson_values()
                else:
                    explainer = explainer_cls(mg, pl_module)
                    my_values = explainer.calculate_all_myerson_values()

                self.mol_idxs.append(i)
                self.explanations.append(my_values)

    def on_predict_end(self, trainer, pl_module):
        model_counter_string = "" if self.max_model_counter == 0 else f"_{self.model_counter}"

        if self.save_as_json:
            file_extension = ".json"
            save_path = (
                self.output_path_dir
                / f"{self.output_filename_base}{model_counter_string}{file_extension}"
            )
            explanations_for_json = []
            for arr in self.explanations:
                if arr.ndim == 2:
                    explanations_for_json.append(arr.T.tolist())
                else:
                    explanations_for_json.append(arr.tolist())
            with open(save_path, "w") as f:
                json.dump(explanations_for_json, f, indent=4)
        else:
            file_extension = ".npz"
            save_path = (
                self.output_path_dir
                / f"{self.output_filename_base}{model_counter_string}{file_extension}"
            )
            np.savez_compressed(save_path, *self.explanations)

        logger.info(f"Myerson explanations saved to {save_path}")
        self.model_counter += 1
