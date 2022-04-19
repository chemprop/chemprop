from typing import List

import torch
from tqdm import tqdm
import numpy as np

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from chemprop.nn_utils import replace_dropout_layers


def predict(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    disable_progress_bar: bool = False,
    scaler: StandardScaler = None,
    return_unc_parameters: bool = False,
    dropout_prob: float = 0.0,
) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param return_unc_parameters: A bool indicating whether additional uncertainty parameters would be returned alongside the mean predictions.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()

    # Replace dropout layers with new dropout layers that work during inference for uncertainty estimation
    if dropout_prob > 0.0:
        replace_dropout_layers(model, dropout_prob)

    preds = []

    var, lambdas, alphas, betas = [], [], [], []  # only used if returning uncertainty parameters

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        (
            mol_batch,
            features_batch,
            atom_descriptors_batch,
            atom_features_batch,
            bond_features_batch,
        ) = (
            batch.batch_graph(),
            batch.features(),
            batch.atom_descriptors(),
            batch.atom_features(),
            batch.bond_features(),
        )

        # Make predictions
        with torch.no_grad():
            batch_preds = model(
                mol_batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_features_batch,
            )

        batch_preds = batch_preds.data.cpu().numpy()

        if model.loss_function == "mve":
            batch_preds, batch_var = np.split(batch_preds, 2, axis=1)
        elif model.loss_function == "evidential":
            if model.classification:
                batch_alphas = np.reshape(
                    batch_preds, [batch_preds.shape[0], batch_preds.shape[1] // 2, 2]
                )
                batch_preds = batch_alphas[:, :, 1] / np.sum(
                    batch_alphas, axis=2
                )  # shape(data, tasks, 2)
            elif model.multiclass:
                batch_alphas = batch_preds
                batch_preds = batch_preds / np.sum(
                    batch_alphas, axis=2, keepdims=True
                )  # shape(data, tasks, num_classes)
            else:  # regression
                batch_preds, batch_lambdas, batch_alphas, batch_betas = np.split(
                    batch_preds, 4, axis=1
                )

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)
            if model.loss_function == "mve":
                batch_var = batch_var * scaler.stds ** 2
            elif model.loss_function == "evidential":
                batch_betas = batch_betas * scaler.stds ** 2

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
        if model.loss_function == "mve":
            var.extend(batch_var.tolist())
        if model.loss_function == "evidential":
            if model.classification:
                alphas.extend(batch_alphas.tolist())
            elif not model.multiclass:  # regression
                lambdas.extend(batch_lambdas.tolist())
                alphas.extend(batch_alphas.tolist())
                betas.extend(batch_betas.tolist())

    if return_unc_parameters:
        if model.loss_function == "mve":
            return preds, var
        if model.loss_function == "evidential":
            if model.classification:
                return preds, alphas
            elif model.multiclass:
                return preds, alphas
            else:
                return preds, lambdas, alphas, betas

    return preds
