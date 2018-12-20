from argparse import Namespace
from typing import List

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.multiprocessing import Process, Queue
import numpy as np
from tqdm import trange

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.features import mol2graph
from chemprop.features.async_featurization import async_mol2graph
from chemprop.models import build_model
from chemprop.utils import get_loss_func


def predict(model: nn.Module,
            data: MoleculeDataset,
            args: Namespace,
            scaler: StandardScaler = None,
            bert_save_memory: bool = False) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param args: Arguments.
    :param scaler: A StandardScaler object fit on the training targets.
    :param bert_save_memory: Store unused predictions as None to avoid unnecessary memory use.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []
    if args.dataset_type == 'bert_pretraining':
        features_preds = []

    if args.maml:
        num_iters, iter_step = data.num_tasks() * args.maml_batches_per_epoch, 1
        full_targets = []
    else:
        num_iters, iter_step = len(data), args.batch_size
    
    if args.parallel_featurization:
        batch_queue = Queue(args.batch_queue_max_size)
        exit_queue = Queue(1)
        batch_process = Process(target=async_mol2graph, args=(batch_queue, data, args, num_iters, iter_step, exit_queue, True))
        batch_process.start()
        currently_loaded_batches = []

    for i in trange(0, num_iters, iter_step):
        if args.maml:
            task_train_data, task_test_data, task_idx = data.sample_maml_task(args, seed=0)
            mol_batch = task_test_data
            smiles_batch, features_batch, targets_batch = task_train_data.smiles(), task_train_data.features(), task_train_data.targets(task_idx)
            targets = torch.Tensor(targets_batch).unsqueeze(1)
            if args.cuda:
                targets = targets.cuda()
        else:
            # Prepare batch
            if args.parallel_featurization:
                if len(currently_loaded_batches) == 0:
                    currently_loaded_batches = batch_queue.get()
                mol_batch, featurized_mol_batch = currently_loaded_batches.pop(0)
            else:
                mol_batch = MoleculeDataset(data[i:i + args.batch_size])
            smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        if args.dataset_type == 'bert_pretraining':
            batch = mol2graph(smiles_batch, args)
            batch.bert_mask(mol_batch.mask())
        else:
            batch = smiles_batch
        
        if args.maml:  # TODO refactor with train loop
            model.zero_grad()
            intermediate_preds = model(batch, features_batch)
            loss = get_loss_func(args)(intermediate_preds, targets)
            loss = loss.sum() / len(batch)
            grad = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
            theta = [p for p in model.named_parameters() if p[1].requires_grad]  # comes in same order as grad
            theta_prime = {p[0]: p[1] - args.maml_lr * grad[i] for i, p in enumerate(theta)}
            for name, nongrad_param in [p for p in model.named_parameters() if not p[1].requires_grad]:
                theta_prime[name] = nongrad_param + torch.zeros(nongrad_param.size()).to(nongrad_param)
            model_prime = build_model(args=args, params=theta_prime)
            smiles_batch, features_batch, targets_batch = task_test_data.smiles(), task_test_data.features(), task_test_data.targets(task_idx)
            # no mask since we only picked data points that have the desired target
            with torch.no_grad():
                batch_preds = model_prime(smiles_batch, features_batch)
            full_targets.extend([[t] for t in targets_batch])
        else:
            with torch.no_grad():
                if args.parallel_featurization:
                    previous_graph_input_mode = model.encoder.graph_input
                    model.encoder.graph_input = True  # force model to accept already processed input
                    batch_preds = model(featurized_mol_batch, features_batch)
                    model.encoder.graph_input = previous_graph_input_mode
                else:
                    batch_preds = model(batch, features_batch)

                if args.dataset_type == 'bert_pretraining':
                    if batch_preds['features'] is not None:
                        features_preds.extend(batch_preds['features'].data.cpu().numpy())
                    batch_preds = batch_preds['vocab']
                
                if args.dataset_type == 'kernel':
                    batch_preds = batch_preds.view(int(batch_preds.size(0)/2), 2, batch_preds.size(1))
                    batch_preds = model.kernel_output_layer(batch_preds)

        batch_preds = batch_preds.data.cpu().numpy()

        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)
        
        if args.dataset_type == 'regression_with_binning':
            batch_preds = batch_preds.reshape((batch_preds.shape[0], args.num_tasks, args.num_bins))
            indices = np.argmax(batch_preds, axis=2)
            preds.extend(indices.tolist())
        else:
            batch_preds = batch_preds.tolist()
            if args.dataset_type == 'bert_pretraining' and bert_save_memory:
                for atom_idx, mask_val in enumerate(mol_batch.mask()):
                    if mask_val != 0:
                        batch_preds[atom_idx] = None  # not going to predict, so save some memory when passing around
            preds.extend(batch_preds)
    
    if args.dataset_type == 'regression_with_binning':
        preds = args.bin_predictions[np.array(preds)].tolist()

    if args.dataset_type == 'bert_pretraining':
        preds = {
            'features': features_preds if len(features_preds) > 0 else None,
            'vocab': preds
        }

    if args.parallel_featurization:
        exit_queue.put(0)  # dummy var to get the subprocess to know that we're done
        batch_process.join()

    if args.maml:
        # return the task targets here to guarantee alignment;
        # there's probably no reasonable scenario where we'd use MAML directly to predict something that's actually unknown
        return preds, full_targets
    return preds
