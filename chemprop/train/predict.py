from typing import List

import numpy as np
import torch
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler, AtomBondScaler
from chemprop.models import MoleculeModel
from chemprop.nn_utils import activate_dropout


def predict(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    disable_progress_bar: bool = False,
    scaler: StandardScaler = None,
    atom_bond_scaler: AtomBondScaler = None,
    return_unc_parameters: bool = False,
    dropout_prob: float = 0.0,
) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param atom_bond_scaler: A :class:`~chemprop.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
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
        bond_descriptors_batch = batch.bond_descriptors()
        bond_features_batch = batch.bond_features()
        constraints_batch = batch.constraints()

        if model.is_atom_bond_targets:
            natoms, nbonds = batch.number_of_atoms, batch.number_of_bonds
            natoms, nbonds = np.array(natoms).flatten(), np.array(nbonds).flatten()
            constraints_batch = np.transpose(constraints_batch).tolist()
            device = next(model.parameters()).device

            # If the path to constraints is not given, the constraints matrix needs to be reformatted.
            if constraints_batch == []:
                for _ in batch._data:
                    natom_targets = len(model.atom_targets)
                    nbond_targets = len(model.bond_targets)
                    ntargets = natom_targets + nbond_targets
                    constraints_batch.append([None] * ntargets)

            ind = 0
            for i in range(len(model.atom_targets)):
                if not model.atom_constraints[i]:
                    constraints_batch[ind] = None
                else:
                    mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                    for j, natom in enumerate(natoms):
                        constraints_batch[ind][j] = (constraints_batch[ind][j] - natom * mean) / std
                    constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(device)
                ind += 1
            for i in range(len(model.bond_targets)):
                if not model.bond_constraints[i]:
                    constraints_batch[ind] = None
                else:
                    mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                    for j, nbond in enumerate(nbonds):
                        constraints_batch[ind][j] = (constraints_batch[ind][j] - nbond * mean) / std
                    constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(device)
                ind += 1
            bond_types_batch = []
            for i in range(len(model.atom_targets)):
                bond_types_batch.append(None)
            for i in range(len(model.bond_targets)):
                if model.adding_bond_types and atom_bond_scaler is not None:
                    mean, std = atom_bond_scaler.means[i+len(model.atom_targets)][0], atom_bond_scaler.stds[i+len(model.atom_targets)][0]
                    bond_types = [(b.GetBondTypeAsDouble() - mean) / std for d in batch for b in d.mol[0].GetBonds()]
                    bond_types = torch.FloatTensor(bond_types).to(device)
                    bond_types_batch.append(bond_types)
                else:
                    bond_types_batch.append(None)
        else:
            bond_types_batch = None

        # Make predictions
        with torch.no_grad():
            batch_preds = model(
                mol_batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
                constraints_batch,
                bond_types_batch,
            )

        if model.is_atom_bond_targets:
            batch_preds = [x.data.cpu().numpy() for x in batch_preds]
            batch_vars, batch_lambdas, batch_alphas, batch_betas = [], [], [], []

            for i, batch_pred in enumerate(batch_preds):
                if model.loss_function == "mve":
                    batch_pred, batch_var = np.split(batch_pred, 2, axis=1)
                    batch_vars.append(batch_var)
                elif model.loss_function == "dirichlet":
                    if model.classification:
                        batch_alpha = np.reshape(
                            batch_pred,
                            [batch_pred.shape[0], batch_pred.shape[1] // 2, 2],
                        )
                        batch_pred = batch_alpha[:, :, 1] / np.sum(
                            batch_alpha, axis=2
                        )  # shape(data, tasks, 2)
                        batch_alphas.append(batch_alpha)
                    elif model.multiclass:
                        raise ValueError(
                            f"In atomic/bond properties prediction, {model.multiclass} is not supported."
                        )
                elif model.loss_function == "evidential":  # regression
                    batch_pred, batch_lambda, batch_alpha, batch_beta = np.split(
                        batch_pred, 4, axis=1
                    )
                    batch_alphas.append(batch_alpha)
                    batch_lambdas.append(batch_lambda)
                    batch_betas.append(batch_beta)
                batch_preds[i] = batch_pred

            # Inverse scale for each atom/bond target if regression
            if atom_bond_scaler is not None:
                batch_preds = atom_bond_scaler.inverse_transform(batch_preds)
                for i, stds in enumerate(atom_bond_scaler.stds):
                    if model.loss_function == "mve":
                        batch_vars[i] = batch_vars[i] * stds ** 2
                    elif model.loss_function == "evidential":
                        batch_betas[i] = batch_betas[i] * stds ** 2

            # Collect vectors
            preds.append(batch_preds)
            if model.loss_function == "mve":
                var.append(batch_vars)
            elif model.loss_function == "dirichlet" and model.classification:
                alphas.append(batch_alphas)
            elif model.loss_function == "evidential":  # regression
                lambdas.append(batch_lambdas)
                alphas.append(batch_alphas)
                betas.append(batch_betas)
        else:
            batch_preds = batch_preds.data.cpu().numpy()

            if model.loss_function == "mve":
                batch_preds, batch_var = np.split(batch_preds, 2, axis=1)
            elif model.loss_function == "dirichlet":
                if model.classification:
                    batch_alphas = np.reshape(
                        batch_preds,
                        [batch_preds.shape[0], batch_preds.shape[1] // 2, 2],
                    )
                    batch_preds = batch_alphas[:, :, 1] / np.sum(
                        batch_alphas, axis=2
                    )  # shape(data, tasks, 2)
                elif model.multiclass:
                    batch_alphas = batch_preds
                    batch_preds = batch_preds / np.sum(
                        batch_alphas, axis=2, keepdims=True
                    )  # shape(data, tasks, num_classes)
            elif model.loss_function == "evidential":  # regression
                batch_preds, batch_lambdas, batch_alphas, batch_betas = np.split(
                    batch_preds, 4, axis=1
                )

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
                if model.loss_function == "mve":
                    batch_var = batch_var * scaler.stds**2
                elif model.loss_function == "evidential":
                    batch_betas = batch_betas * scaler.stds**2

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
        var = [np.concatenate(x) for x in zip(*var)]
        alphas = [np.concatenate(x) for x in zip(*alphas)]
        betas = [np.concatenate(x) for x in zip(*betas)]
        lambdas = [np.concatenate(x) for x in zip(*lambdas)]

    if return_unc_parameters:
        if model.loss_function == "mve":
            return preds, var
        elif model.loss_function == "dirichlet":
            return preds, alphas
        elif model.loss_function == "evidential":
            return preds, lambdas, alphas, betas

    return preds
