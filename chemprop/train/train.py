import logging
from typing import Callable, List

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: MoleculeModel,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          atom_bond_scalers: List[StandardScaler] = None,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param atom_bond_scalers: A list of :class:`~chemprop.data.scaler.StandardScaler` fitted on each atomic/bond target.
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
        mol_batch, features_batch, target_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, data_weights_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_features(), batch.data_weights()

        if model.is_atom_bond_targets:
            masks, targets = [], []
            for tb in zip(*target_batch):
                tb = np.concatenate(tb)
                masks.append(torch.tensor([x is not None for x in tb], dtype=torch.bool))
                targets.append(torch.tensor([0 if x is None else x for x in tb], dtype=torch.float))
            if args.target_weights is not None:
                target_weights = [torch.ones(1, 1) * i for i in args.target_weights] # shape(tasks,1)
            else:
                target_weights = [torch.ones(1, 1) for i in targets]
            data_weights = batch.atom_bond_data_weights()
            data_weights = [torch.tensor(x).unsqueeze(1) for x in data_weights]

            natoms, nbonds = batch.number_of_atoms, batch.number_of_bonds
            constraints_batch = []
            ind = 0
            for i in range(len(args.atom_targets)):
                if args.atom_constraints is None:
                    constraints_batch.append(None)
                elif i < len(args.atom_constraints):
                    mean, std = atom_bond_scalers[ind].means[0], atom_bond_scalers[ind].stds[0]
                    constraints = torch.tensor([(args.atom_constraints[i] - natom * mean) / std for natom in natoms])
                    constraints_batch.append(constraints.to(args.device))
                else:
                    constraints_batch.append(None)
                ind += 1
            for i in range(len(args.bond_targets)):
                if args.bond_constraints is None:
                    constraints_batch.append(None)
                elif i < len(args.bond_constraints):
                    mean, std = atom_bond_scalers[ind].means[0], atom_bond_scalers[ind].stds[0]
                    constraints = torch.tensor([(args.bond_constraints[i] - nbond * mean) / std for nbond in nbonds])
                    constraints_batch.append(constraints.to(args.device))
                else:
                    constraints_batch.append(None)
                ind += 1
        else:
            masks = torch.tensor([[x is not None for x in tb] for tb in target_batch], dtype=torch.bool) # shape(batch, tasks)
            targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch]) # shape(batch, tasks)

            if args.target_weights is not None:
                target_weights = torch.tensor(args.target_weights).unsqueeze(0) # shape(1,tasks)
            else:
                target_weights = torch.ones(targets.shape[1]).unsqueeze(0)
            data_weights = torch.tensor(data_weights_batch).unsqueeze(1) # shape(batch,1)

            constraints_batch = None

            if args.loss_function == 'bounded_mse':
                lt_target_batch = batch.lt_targets() # shape(batch, tasks)
                gt_target_batch = batch.gt_targets() # shape(batch, tasks)
                lt_target_batch = torch.tensor(lt_target_batch)
                gt_target_batch = torch.tensor(gt_target_batch)

        # Run model
        model.zero_grad()
        preds = model(mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, constraints_batch)

        # Move tensors to correct device
        torch_device = preds.device
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
            if args.loss_function == 'bounded_mse':
                lt_target_batch = lt_target_batch.to(torch_device)
                gt_target_batch = gt_target_batch.to(torch_device)

        # Calculate losses
        if model.is_atom_bond_targets:
            loss_multi_task = []
            for target, pred, target_weight, data_weight, mask in zip(targets, preds, target_weights, data_weights, masks):
                if args.loss_function == 'mcc' and args.dataset_type == 'classification':
                    loss = loss_func(pred, target, data_weight, mask) * target_weight.squeeze(0)
                elif args.loss_function == 'bounded_mse':
                    raise ValueError(f'Loss function "{args.loss_function}" is not supported with dataset type {args.dataset_type} in atomic/bond properties prediction.')
                elif args.loss_function in ['binary_cross_entropy', 'mse']:
                    loss = loss_func(pred, target) * target_weight * data_weight * mask
                elif args.loss_function in ['evidential', 'dirichlet']:
                    loss = loss_func(preds, target, args.evidential_regularization) * target_weight * data_weight * mask
                else:
                    raise ValueError(f'Dataset type "{args.dataset_type}" is not supported.')
                loss = loss.sum() / mask.sum()
                loss_multi_task.append(loss)

            loss_sum = [x + y for x, y in zip(loss_sum, loss_multi_task)]
            iter_count += 1

            sum(loss_multi_task).backward()
        else:
            if args.loss_function == 'mcc' and args.dataset_type == 'classification':
                loss = loss_func(preds, targets, data_weights, masks) * target_weights.squeeze(0)
            elif args.loss_function == 'mcc': # multiclass dataset type
                targets = targets.long()
                target_losses = []
                for target_index in range(preds.size(1)):
                    target_loss = loss_func(preds[:, target_index, :], targets[:, target_index], data_weights, masks[:, target_index]).unsqueeze(0)
                    target_losses.append(target_loss)
                loss = torch.cat(target_losses).to(torch_device) * target_weights.squeeze(0)
            elif args.dataset_type == 'multiclass':
                targets = targets.long()
                if args.loss_function == 'dirichlet':
                    loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * mask
                else:
                    target_losses = []
                    for target_index in range(preds.size(1)):
                        target_loss = loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1)
                        target_losses.append(target_loss)
                    loss = torch.cat(target_losses, dim=1).to(torch_device) * target_weights * data_weights * mask
            elif args.dataset_type == 'spectra':
                loss = loss_func(preds, targets, masks) * target_weights * data_weights * masks
            elif args.loss_function == 'bounded_mse':
                loss = loss_func(preds, targets, lt_target_batch, gt_target_batch) * target_weights * data_weights * masks
            elif args.loss_function == 'evidential':
                loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * mask
            elif args.loss_function == 'dirichlet': # classification
                loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * mask
            else:
                loss = loss_func(preds, targets) * target_weights * data_weights * masks

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

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
