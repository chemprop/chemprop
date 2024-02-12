import logging
from typing import Callable

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset, AtomBondScaler
from chemprop.models import MoleculeModel
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    loss_func: Callable,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    args: TrainArgs,
    n_iter: int = 0,
    atom_bond_scaler: AtomBondScaler = None,
    logger: logging.Logger = None,
    writer: SummaryWriter = None,
) -> int:
    """
    Trains a model for an epoch.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param atom_bond_scaler: A :class:`~chemprop.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param logger: A logger for recording output.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    if model.is_atom_bond_targets:
        loss_sum, iter_count = [0]*(len(args.atom_targets) + len(args.bond_targets)), 0
    else:
        loss_sum = iter_count = 0

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch, mask_batch, atom_descriptors_batch, atom_features_batch, bond_descriptors_batch, bond_features_batch, constraints_batch, data_weights_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.mask(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_descriptors(), batch.bond_features(), batch.constraints(), batch.data_weights()

        if model.is_atom_bond_targets:
            targets = []
            for dt in zip(*target_batch):
                dt = np.concatenate(dt)
                targets.append(torch.tensor([0 if x is None else x for x in dt], dtype=torch.float))
            masks = [torch.tensor(mask, dtype=torch.bool) for mask in mask_batch]
            if args.target_weights is not None:
                target_weights = [torch.ones(1, 1) * i for i in args.target_weights]  # shape(tasks, 1)
            else:
                target_weights = [torch.ones(1, 1) for i in targets]
            data_weights = batch.atom_bond_data_weights()
            data_weights = [torch.tensor(x).unsqueeze(1) for x in data_weights]

            natoms, nbonds = batch.number_of_atoms, batch.number_of_bonds
            natoms, nbonds = np.array(natoms).flatten(), np.array(nbonds).flatten()
            constraints_batch = np.transpose(constraints_batch).tolist()
            ind = 0
            for i in range(len(args.atom_targets)):
                if not args.atom_constraints[i]:
                    constraints_batch[ind] = None
                else:
                    mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                    for j, natom in enumerate(natoms):
                        constraints_batch[ind][j] = (constraints_batch[ind][j] - natom * mean) / std
                    constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(args.device)
                ind += 1
            for i in range(len(args.bond_targets)):
                if not args.bond_constraints[i]:
                    constraints_batch[ind] = None
                else:
                    mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                    for j, nbond in enumerate(nbonds):
                        constraints_batch[ind][j] = (constraints_batch[ind][j] - nbond * mean) / std
                    constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(args.device)
                ind += 1
            bond_types_batch = []
            for i in range(len(args.atom_targets)):
                bond_types_batch.append(None)
            for i in range(len(args.bond_targets)):
                if args.adding_bond_types and atom_bond_scaler is not None:
                    mean, std = atom_bond_scaler.means[i+len(args.atom_targets)][0], atom_bond_scaler.stds[i+len(args.atom_targets)][0]
                    bond_types = [(b.GetBondTypeAsDouble() - mean) / std for d in batch for b in d.mol[0].GetBonds()]
                    bond_types = torch.FloatTensor(bond_types).to(args.device)
                    bond_types_batch.append(bond_types)
                else:
                    bond_types_batch.append(None)
        else:
            mask_batch = np.transpose(mask_batch).tolist()
            masks = torch.tensor(mask_batch, dtype=torch.bool)  # shape(batch, tasks)
            targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch])  # shape(batch, tasks)

            if args.target_weights is not None:
                target_weights = torch.tensor(args.target_weights).unsqueeze(0)  # shape(1,tasks)
            else:
                target_weights = torch.ones(targets.shape[1]).unsqueeze(0)
            data_weights = torch.tensor(data_weights_batch).unsqueeze(1)  # shape(batch,1)

            constraints_batch = None
            bond_types_batch = None

            if args.loss_function == "bounded_mse":
                lt_target_batch = batch.lt_targets()  # shape(batch, tasks)
                gt_target_batch = batch.gt_targets()  # shape(batch, tasks)
                lt_target_batch = torch.tensor(lt_target_batch)
                gt_target_batch = torch.tensor(gt_target_batch)

        # Run model
        model.zero_grad()
        preds = model(
            mol_batch,
            features_batch,
            atom_descriptors_batch,
            atom_features_batch,
            bond_descriptors_batch,
            bond_features_batch,
            constraints_batch,
            bond_types_batch,
        )

        # Move tensors to correct device
        torch_device = args.device
        if model.is_atom_bond_targets:
            masks = [x.to(torch_device) for x in masks]
            masks = [x.reshape([-1, 1]) for x in masks]
            targets = [x.to(torch_device) for x in targets]
            targets = [x.reshape([-1, 1]) for x in targets]
            target_weights = [x.to(torch_device) for x in target_weights]
            data_weights = [x.to(torch_device) for x in data_weights]
        else:
            masks = masks.to(torch_device)
            targets = targets.to(torch_device)
            target_weights = target_weights.to(torch_device)
            data_weights = data_weights.to(torch_device)
            if args.loss_function == "bounded_mse":
                lt_target_batch = lt_target_batch.to(torch_device)
                gt_target_batch = gt_target_batch.to(torch_device)

        # Calculate losses
        if model.is_atom_bond_targets:
            loss_multi_task = []
            for target, pred, target_weight, data_weight, mask in zip(targets, preds, target_weights, data_weights, masks):
                if args.loss_function == "mcc" and args.dataset_type == "classification":
                    loss = loss_func(pred, target, data_weight, mask) * target_weight.squeeze(0)
                elif args.loss_function == "bounded_mse":
                    raise ValueError(f'Loss function "{args.loss_function}" is not supported with dataset type {args.dataset_type} in atomic/bond properties prediction.')
                elif args.loss_function in ["binary_cross_entropy", "mse", "mve"]:
                    loss = loss_func(pred, target) * target_weight * data_weight * mask
                elif args.loss_function == "evidential":
                    loss = loss_func(pred, target, args.evidential_regularization) * target_weight * data_weight * mask
                elif args.loss_function == "dirichlet" and args.dataset_type == "classification":
                    loss = loss_func(pred, target, args.evidential_regularization) * target_weight * data_weight * mask
                elif args.loss_function == "quantile_interval":
                    quantiles_tensor = torch.tensor(args.quantiles, device=torch_device)
                    loss = loss_func(pred, target, quantiles_tensor) * target_weight * data_weight * mask
                else:
                    raise ValueError(f'Dataset type "{args.dataset_type}" is not supported.')
                loss = loss.sum() / mask.sum()
                loss_multi_task.append(loss)

            loss_sum = [x + y for x, y in zip(loss_sum, loss_multi_task)]
            iter_count += 1

            sum(loss_multi_task).backward()
        else:
            if args.loss_function == "mcc" and args.dataset_type == "classification":
                loss = loss_func(preds, targets, data_weights, masks) * target_weights.squeeze(0)
            elif args.loss_function == "mcc":  # multiclass dataset type
                targets = targets.long()
                target_losses = []
                for target_index in range(preds.size(1)):
                    target_loss = loss_func(preds[:, target_index, :], targets[:, target_index], data_weights, masks[:, target_index]).unsqueeze(0)
                    target_losses.append(target_loss)
                loss = torch.cat(target_losses) * target_weights.squeeze(0)
            elif args.dataset_type == "multiclass":
                targets = targets.long()
                if args.loss_function == "dirichlet":
                    loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * masks
                else:
                    target_losses = []
                    for target_index in range(preds.size(1)):
                        target_loss = loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1)
                        target_losses.append(target_loss)
                    loss = torch.cat(target_losses, dim=1).to(torch_device) * target_weights * data_weights * masks
            elif args.dataset_type == "spectra":
                loss = loss_func(preds, targets, masks) * target_weights * data_weights * masks
            elif args.loss_function == "bounded_mse":
                loss = loss_func(preds, targets, lt_target_batch, gt_target_batch) * target_weights * data_weights * masks
            elif args.loss_function == "evidential":
                loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * masks
            elif args.loss_function == "dirichlet":  # classification
                loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * masks
            elif args.loss_function == "quantile_interval":
                quantiles_tensor = torch.tensor(args.quantiles, device=torch_device)
                loss = loss_func(preds, targets, quantiles_tensor) * target_weights * data_weights * masks
            else:
                loss = loss_func(preds, targets) * target_weights * data_weights * masks


            if args.loss_function == "mcc":
                loss = loss.mean()
            else:
                loss = loss.sum() / masks.sum()

            loss_sum += loss.item()
            iter_count += 1

            loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            if model.is_atom_bond_targets:
                loss_avg = sum(loss_sum) / iter_count
                loss_sum, iter_count = [0]*(len(args.atom_targets) + len(args.bond_targets)), 0
            else:
                loss_avg = loss_sum / iter_count
                loss_sum = iter_count = 0

            lrs_str = ", ".join(f"lr_{i} = {lr:.4e}" for i, lr in enumerate(lrs))
            debug(f"Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}")

            if writer is not None:
                writer.add_scalar("train_loss", loss_avg, n_iter)
                writer.add_scalar("param_norm", pnorm, n_iter)
                writer.add_scalar("gradient_norm", gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f"learning_rate_{i}", lr, n_iter)

    return n_iter
