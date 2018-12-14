from argparse import Namespace
from typing import List

import torch
import torch.nn as nn
import numpy as np

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.features import mol2graph
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
    for i in range(0, num_iters, iter_step):
        if args.maml:
            task_train_data, task_test_data, task_idx = data.sample_maml_task(args, seed=0)
            mol_batch = task_test_data
            smiles_batch, features_batch, targets_batch = task_train_data.smiles(), task_train_data.features(), task_train_data.targets(task_idx)
            targets = torch.Tensor(targets_batch).unsqueeze(1)
            if args.cuda:
                targets = targets.cuda()
        else:
            # Prepare batch
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
            loss.backward()
            params = {name: param for name, param in model.named_parameters()}
            for name in params.keys():
                if params[name].grad is None:
                    params[name] = params[name] + torch.zeros(params[name].size()).to(params[name])
                else:
                    params[name] = params[name] - args.maml_lr * params[name].grad.data
            model_prime = build_model(args=args, params=params)
            smiles_batch, features_batch, targets_batch = task_test_data.smiles(), task_test_data.features(), task_test_data.targets(task_idx)
            # no mask since we only picked data points that have the desired target
            model_prime.zero_grad()
            batch_preds = model_prime(smiles_batch, features_batch)
            full_targets.extend([[t] for t in targets_batch])
        else:
            with torch.no_grad():
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

    if args.maml:
        # return the task targets here to guarantee alignment;
        # there's probably no reasonable scenario where we'd use MAML directly to predict something that's actually unknown
        return preds, full_targets
    return preds
