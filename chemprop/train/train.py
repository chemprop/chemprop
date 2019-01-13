from argparse import Namespace
import logging
import random
from typing import Callable, List, Union
import os

from tensorboardX import SummaryWriter
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR
from tqdm import trange, tqdm
import pickle
from copy import deepcopy
from torch.multiprocessing import Process, Queue

from chemprop.data import MoleculeDataset
from chemprop.features import featurization, mol2graph
from chemprop.features.async_featurization import async_mol2graph
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR
from chemprop.models import build_model


def train(model: nn.Module,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None,
          chunk_names: bool = False,
          val_smiles: List[str] = None,
          test_smiles: List[str] = None) -> int:
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
    :param chunk_names: Whether to train on the data in chunks. In this case,
    data must be a list of paths to the data chunks.
    :param val_smiles: Validation smiles strings without targets.
    :param test_smiles: Test smiles strings without targets, used for adversarial setting.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    
    model.train()

    if args.dataset_type == 'bert_pretraining':
        features_loss = nn.MSELoss()

    if chunk_names:
        for path, memo_path in tqdm(data, total=len(data)):
            featurization.SMILES_TO_FEATURES = dict()
            if os.path.isfile(memo_path):
                found_memo = True
                with open(memo_path, 'rb') as f:
                    featurization.SMILES_TO_FEATURES = pickle.load(f)
            else:
                found_memo = False
            with open(path, 'rb') as f:
                chunk = pickle.load(f)
            if args.moe:
                for source in chunk:
                    source.shuffle()
            else:
                chunk.shuffle()
            n_iter = train(
                model=model,
                data=chunk,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer,
                chunk_names=False,
                val_smiles=val_smiles,
                test_smiles=test_smiles
            )
            if not found_memo:
                with open(memo_path, 'wb') as f:
                    pickle.dump(featurization.SMILES_TO_GRAPH, f, protocol=pickle.HIGHEST_PROTOCOL)
        return n_iter
    
    if not args.moe:
        data.shuffle()

    loss_sum, iter_count = 0, 0
    if args.adversarial:
        if args.moe:
            train_smiles = []
            for d in data:
                train_smiles += d.smiles()
        else:
            train_smiles = data.smiles()
        train_val_smiles = train_smiles + val_smiles
        d_loss_sum, g_loss_sum, gp_norm_sum = 0, 0, 0

    if args.moe:
        test_smiles = list(test_smiles)
        random.shuffle(test_smiles)
        train_smiles = []
        for d in data:
            d.shuffle()
            train_smiles.append(d.smiles())
        num_iters = min(len(test_smiles), min([len(d) for d in data]))
    elif args.maml:
        num_iters = args.maml_batches_per_epoch * args.maml_batch_size
        model.zero_grad()
        maml_sum_loss = 0
    else:
        num_iters = len(data) if args.last_batch else len(data) // args.batch_size * args.batch_size

    if args.parallel_featurization:
        batch_queue = Queue(args.batch_queue_max_size)
        exit_queue = Queue(1)
        batch_process = Process(target=async_mol2graph, args=(batch_queue, data, args, num_iters, args.batch_size, exit_queue, args.last_batch))
        batch_process.start()
        currently_loaded_batches = []

    iter_size = 1 if args.maml else args.batch_size

    for i in trange(0, num_iters, iter_size):
        if args.moe:
            if not args.batch_domain_encs:
                model.compute_domain_encs(train_smiles)  # want to recompute every batch
            mol_batch = [MoleculeDataset(d[i:i + args.batch_size]) for d in data]
            train_batch, train_targets = [], []
            for b in mol_batch:
                tb, tt = b.smiles(), b.targets()
                train_batch.append(tb)
                train_targets.append(tt)
            test_batch = test_smiles[i:i + args.batch_size]
            loss = model.compute_loss(train_batch, train_targets, test_batch)
            model.zero_grad()

            loss_sum += loss.item()
            iter_count += len(mol_batch)
        elif args.maml:
            task_train_data, task_test_data, task_idx = data.sample_maml_task(args)
            mol_batch = task_test_data
            smiles_batch, features_batch, target_batch = task_train_data.smiles(), task_train_data.features(), task_train_data.targets(task_idx)
            # no mask since we only picked data points that have the desired target
            targets = torch.Tensor(target_batch).unsqueeze(1)
            if next(model.parameters()).is_cuda:
                targets = targets.cuda()
            preds = model(smiles_batch, features_batch)
            loss = loss_func(preds, targets)
            loss = loss.sum() / len(smiles_batch)
            grad = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
            theta = [p for p in model.named_parameters() if p[1].requires_grad]  # comes in same order as grad
            theta_prime = {p[0]: p[1] - args.maml_lr * grad[i] for i, p in enumerate(theta)}
            for name, nongrad_param in [p for p in model.named_parameters() if not p[1].requires_grad]:
                theta_prime[name] = nongrad_param + torch.zeros(nongrad_param.size()).to(nongrad_param)
        else:
            # Prepare batch
            if args.parallel_featurization:
                if len(currently_loaded_batches) == 0:
                    currently_loaded_batches = batch_queue.get()
                mol_batch, featurized_mol_batch = currently_loaded_batches.pop()
            else:
                if not args.last_batch and i + args.batch_size > len(data):
                    break
                mol_batch = MoleculeDataset(data[i:i + args.batch_size])
            smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()

            if args.dataset_type == 'bert_pretraining':
                batch = mol2graph(smiles_batch, args)
                mask = mol_batch.mask()
                batch.bert_mask(mask)
                mask = 1 - torch.FloatTensor(mask)  # num_atoms
                features_targets = torch.FloatTensor(target_batch['features']) if target_batch['features'] is not None else None  # num_molecules x features_size
                targets = torch.FloatTensor(target_batch['vocab'])  # num_atoms
                if args.bert_vocab_func == 'feature_vector':
                    mask = mask.reshape(-1, 1)
                else:
                    targets = targets.long()
            else:
                batch = smiles_batch
                mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
                targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

            if next(model.parameters()).is_cuda:
                mask, targets = mask.cuda(), targets.cuda()

                if args.dataset_type == 'bert_pretraining' and features_targets is not None:
                    features_targets = features_targets.cuda()

            if args.class_balance:
                class_weights = []
                for task_num in range(data.num_tasks()):
                    class_weights.append(args.class_weights[task_num][targets[:, task_num].long()])
                class_weights = torch.stack(class_weights).t()  # num_molecules x num_tasks
            else:
                class_weights = torch.ones(targets.shape)

            if args.cuda:
                class_weights = class_weights.cuda()

            # Run model
            model.zero_grad()
            if args.parallel_featurization:
                previous_graph_input_mode = model.encoder.graph_input
                model.encoder.graph_input = True  # force model to accept already processed input
                preds = model(featurized_mol_batch, features_batch)
                model.encoder.graph_input = previous_graph_input_mode
            else:
                preds = model(batch, features_batch)
            if args.dataset_type == 'regression_with_binning':
                preds = preds.view(targets.size(0), targets.size(1), -1)
                targets = targets.long()
                loss = 0
                for task in range(targets.size(1)):
                    loss += loss_func(preds[:, task, :], targets[:, task]) * class_weights[:, task] * mask[:, task]  # for some reason cross entropy doesn't support multi target
                loss = loss.sum() / mask.sum()
            else:
                if args.dataset_type == 'unsupervised':
                    targets = targets.long().reshape(-1)

                if args.dataset_type == 'bert_pretraining':
                    features_preds, preds = preds['features'], preds['vocab']
                
                if args.dataset_type == 'kernel':
                    preds = preds.view(int(preds.size(0)/2), 2, preds.size(1))
                    preds = model.kernel_output_layer(preds)

                loss = loss_func(preds, targets) * class_weights * mask
                if args.predict_features_and_task:
                    loss = (loss.sum() + loss[:, :-args.features_size].sum() * (args.task_weight-1)) \
                                / (mask.sum() + mask[:, :-args.features_size].sum() * (args.task_weight-1))
                else:
                    loss = loss.sum() / mask.sum()

                if args.dataset_type == 'bert_pretraining' and features_targets is not None:
                    loss += features_loss(features_preds, features_targets)

            loss_sum += loss.item()
            iter_count += len(mol_batch)

        if args.maml:
            model_prime = build_model(args=args, params=theta_prime)
            smiles_batch, features_batch, target_batch = task_test_data.smiles(), task_test_data.features(), [t[task_idx] for t in task_test_data.targets()]
            # no mask since we only picked data points that have the desired target
            targets = torch.Tensor([[t] for t in target_batch])
            if next(model_prime.parameters()).is_cuda:
                targets = targets.cuda()
            model_prime.zero_grad()
            preds = model_prime(smiles_batch, features_batch)
            loss = loss_func(preds, targets)
            loss = loss.sum() / len(smiles_batch)
            loss_sum += loss.item()
            iter_count += len(smiles_batch)  # TODO check that this makes sense, but it's just for display
            maml_sum_loss += loss
            if i % args.maml_batch_size == args.maml_batch_size - 1:
                maml_sum_loss.backward()
                optimizer.step()
                model.zero_grad()
                maml_sum_loss = 0
        else:
            loss.backward()
            if args.max_grad_norm is not None:
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.adjust_weight_decay:
            current_pnorm = compute_pnorm(model)
            if current_pnorm < args.pnorm_target:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['weight_decay'] = max(0, optimizer.param_groups[i]['weight_decay'] - args.adjust_weight_decay_step)
            else:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['weight_decay'] += args.adjust_weight_decay_step

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        if args.adversarial:
            for _ in range(args.gan_d_per_g):
                train_val_smiles_batch = random.sample(train_val_smiles, args.batch_size)
                test_smiles_batch = random.sample(test_smiles, args.batch_size)
                d_loss, gp_norm = model.train_D(train_val_smiles_batch, test_smiles_batch)
            train_val_smiles_batch = random.sample(train_val_smiles, args.batch_size)
            test_smiles_batch = random.sample(test_smiles, args.batch_size)
            g_loss = model.train_G(train_val_smiles_batch, test_smiles_batch)

            # we probably only care about the g_loss honestly
            d_loss_sum += d_loss * args.batch_size
            gp_norm_sum += gp_norm * args.batch_size
            g_loss_sum += g_loss * args.batch_size

        n_iter += len(mol_batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            if args.adversarial:
                d_loss_avg, g_loss_avg, gp_norm_avg = d_loss_sum / iter_count, g_loss_sum / iter_count, gp_norm_sum / iter_count
                d_loss_sum, g_loss_sum, gp_norm_sum = 0, 0, 0
            loss_sum, iter_count = 0, 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')
            if args.adversarial:
                debug(f'D Loss = {d_loss_avg:.4e}, G Loss = {g_loss_avg:.4e}, GP Norm = {gp_norm_avg:.4}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    if args.parallel_featurization:
        exit_queue.put(0)  # dummy var to get the subprocess to know that we're done
        batch_process.join()

    return n_iter