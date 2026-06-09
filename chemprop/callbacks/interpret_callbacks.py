import logging
import pickle

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

    The explanations are saved as a pickle file containing a dictionary with the keys ``myerson_values`` and ``sampled``.
    The ``myerson_values`` will be list of 1D or 2D arrays of shape ``num_mols`` (for regression / binary classication)
    or ``num_mols x num_classes`` (multilabel binary classification) containing the explanations.

    Parameters
    ----------
    cli_args : Namespace
        All `predict` command line arguments.
    sampling_threshold : int, default=20
        Maximum number of nodes in a molecule before switching to the sampling explainer.
    """

    def __init__(self, cli_args, sampling_threshold: int = 20):
        super().__init__()
        self.sampling_threshold = sampling_threshold

        model_paths = find_models(cli_args.model_paths)

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

        self.output_filename_base = cli_args.output.stem + "_myerson_explanation"
        self.output_path_dir = cli_args.output.parent

    def on_predict_start(self, trainer, pl_module):
        if f"{pl_module.predictor.__class__.__name__}" not in [
            "BinaryClassificationFFN",
            "RegressionFFN",
        ]:
            raise NotImplementedError(
                f"Myerson explanations are only implemented for BinaryClassificationFNN and RegressionFFN. Got {pl_module.predictor.__class__.__name__}"
            )
        self.mol_idxs = []
        self.per_mol_atom_idxs = []
        self.sampled = []
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
        outputs: any,
        batch: any,
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
                    xai_type = "sampled"
                else:
                    explainer = explainer_cls(mg, pl_module)
                    my_values = explainer.calculate_all_myerson_values()
                    xai_type = "exact"

                self.mol_idxs.append(i)
                self.sampled.append([xai_type == "sampled"])
                self.explanations.append(my_values)

    def on_predict_end(self, trainer, pl_module):
        model_counter_string = "" if self.max_model_counter == 0 else f"_{self.model_counter}"
        save_path = self.output_path_dir / f"{self.output_filename_base}{model_counter_string}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "myerson_values": self.explanations,
                    "sampled": np.array(self.sampled, dtype=bool),
                },
                f,
            )
        logger.info(f"Myerson explanations to {save_path}")
        self.model_counter += 1
