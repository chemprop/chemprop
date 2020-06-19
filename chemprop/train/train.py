import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: nn.Module,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    loss_sum, iter_count = 0, 0
    main_loss_sum, auxiliary_loss_sum = 0, 0

    for batch in tqdm(data_loader, total=len(data_loader)):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch, target_features_batch = batch.batch_graph(), batch.features(), batch.targets(), batch.target_features()
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        # Run model
        model.zero_grad()
        preds, output_dict = model(mol_batch, features_batch)

        # Move tensors to correct device
        mask = mask.to(preds.device)
        targets = targets.to(preds.device)
        class_weights = torch.ones(targets.shape, device=preds.device)
        if model.use_auxiliary:
            target_features_batch = torch.from_numpy(np.stack(target_features_batch)).float().to(preds.device)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            main_loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            main_loss = loss_func(preds, targets) * class_weights * mask

        main_loss = main_loss.sum() / mask.sum()

        if model.use_auxiliary:
            mse_loss = torch.nn.MSELoss(reduction = 'none')
            auxiliary_loss = mse_loss(output_dict['auxiliary'], target_features_batch).mean(axis=1).unsqueeze(1) * mask

            auxiliary_loss = auxiliary_loss.sum() / mask.sum()
        else:
            auxiliary_loss = 0

        loss = main_loss + args.auxiliary_lambda * auxiliary_loss

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        iter_count += len(batch)

        if model.use_auxiliary:
            main_loss_sum += main_loss.sum()
            auxiliary_loss_sum += auxiliary_loss.sum()


        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            main_loss_avg = main_loss_sum / iter_count
            auxiliary_loss_avg = main_loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            aux_string = f'Main loss = {main_loss_avg:.4f}, Aux loss = {auxiliary_loss_avg:.4f}, ' if model.use_auxiliary else ''
            debug(f'Loss = {loss_avg:.4e}, {aux_string}PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                if model.use_auxiliary:
                    writer.add_scalar('main_loss', main_loss_avg, n_iter)
                    writer.add_scalar('aux_loss', auxiliary_loss_avg, n_iter)

                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
