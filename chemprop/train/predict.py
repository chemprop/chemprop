from typing import List

import numpy as np
import torch
from tqdm import tqdm
import numpy as np

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from chemprop.nn_utils import activate_dropout


def predict(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    disable_progress_bar: bool = False,
    scaler: StandardScaler = None,
    atom_bond_scalers: List[StandardScaler] = None,
    return_unc_parameters: bool = False,
    dropout_prob: float = 0.0,
) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param atom_bond_scalers: A list of :class:`~chemprop.data.scaler.StandardScaler` fitted on each atomic/bond target.
    :param return_unc_parameters: A bool indicating whether additional uncertainty parameters would be returned alongside the mean predictions.
    :param dropout_prob: For use during uncertainty prediction only. The propout probability used in generating a dropout ensemble.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks. If returning uncertainty parameters as well,
        it is a tuple of lists of lists, of a length depending on how many uncertainty parameters are appropriate for the loss function.
    """
    model.eval()
    
    # Activate dropout layers to work during inference for uncertainty estimation
    if dropout_prob > 0.0:
        def activate_dropout_(model):
            return activate_dropout(model, dropout_prob)
        model.apply(activate_dropout_)

    preds = []

    var, lambdas, alphas, betas = [], [], [], []  # only used if returning uncertainty parameters

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch = batch.batch_graph()
        features_batch = batch.features()
        atom_descriptors_batch = batch.atom_descriptors()
        atom_features_batch = batch.atom_features()
        bond_features_batch = batch.bond_features()

        if model.is_atom_bond_targets:
            natoms, nbonds = batch.number_of_atoms, batch.number_of_bonds
            constraints_batch = []
            ind = 0
            for i in range(len(model.atom_targets)):
                if model.atom_constraints is None:
                    constraints_batch.append(None)
                elif i < len(model.atom_constraints):
                    mean, std = atom_bond_scalers[ind].means[0], atom_bond_scalers[ind].stds[0]
                    constraints = torch.tensor([(model.atom_constraints[i] - natom * mean) / std for natom in natoms])
                    constraints_batch.append(constraints.to(model.device))
                else:
                    constraints_batch.append(None)
                ind += 1
            for i in range(len(model.bond_targets)):
                if model.bond_constraints is None:
                    constraints_batch.append(None)
                elif i < len(model.bond_constraints):
                    mean, std = atom_bond_scalers[ind].means[0], atom_bond_scalers[ind].stds[0]
                    constraints = torch.tensor([(model.bond_constraints[i] - nbond * mean) / std for nbond in nbonds])
                    constraints_batch.append(constraints.to(model.device))
                else:
                    constraints_batch.append(None)
                ind += 1

        # Make predictions
        with torch.no_grad():
            batch_preds = model(
                mol_batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_features_batch,
                constraints_batch,
            )

        if model.is_atom_bond_targets:
            batch_preds = [x.data.cpu().numpy() for x in batch_preds]

            # Inverse scale for each atom/bond target if regression
            if atom_bond_scalers is not None:
                for i, pred in enumerate(batch_preds):
                    batch_preds[i] = atom_bond_scalers[i].inverse_transform(pred)

            # Collect vectors
            preds.append(batch_preds)
        else:
            batch_preds = batch_preds.data.cpu().numpy()

            if model.loss_function == "mve":
                batch_preds, batch_var = np.split(batch_preds, 2, axis=1)
            elif model.loss_function == "dirichlet":
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
            elif model.loss_function == 'evidential':  # regression
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
            elif model.loss_function == "dirichlet" and model.classification:
                alphas.extend(batch_alphas.tolist())
            elif model.loss_function == "evidential":  # regression
                lambdas.extend(batch_lambdas.tolist())
                alphas.extend(batch_alphas.tolist())
                betas.extend(batch_betas.tolist())
    
    if model.is_atom_bond_targets:
        preds = [np.concatenate(x) for x in zip(*preds)]

    if return_unc_parameters:
        if model.loss_function == "mve":
            return preds, var
        elif model.loss_function == "dirichlet":
            return preds, alphas
        elif model.loss_function == "evidential":
            return preds, lambdas, alphas, betas

    return preds
