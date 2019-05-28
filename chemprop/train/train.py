from argparse import Namespace
import logging
from typing import Callable, List, Union

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange

from chemprop.data import MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: nn.Module,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
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
    
    data.shuffle()

    loss_sum, iter_count = 0, 0

    num_iters = len(data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability

    iter_size = args.batch_size

    for i in trange(0, num_iters, iter_size):
        # Prepare batch
        if i + args.batch_size > len(data):
            break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = smiles_batch
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()

        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        preds = model(batch, features_batch)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += len(mol_batch)

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(mol_batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
