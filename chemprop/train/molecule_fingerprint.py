import csv
from typing import List, Optional, Union

import torch
import numpy as np
from tqdm import tqdm

from chemprop.args import FingerprintArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, makedirs, timeit, load_scalers, update_prediction_args
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.features import set_reaction, set_explicit_h, set_adding_hs, set_keeping_atom_map, reset_featurization_parameters, set_extra_atom_fdim, set_extra_bond_fdim
from chemprop.models import MoleculeModel

@timeit()
def molecule_fingerprint(args: FingerprintArgs,
                         smiles: List[List[str]] = None,
                         return_invalid_smiles: bool = True) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to encode fingerprint vectors for the data.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :param return_invalid_smiles: Whether to return predictions of "Invalid SMILES" for invalid SMILES, otherwise will skip them in returned predictions.
    :return: A list of fingerprint vectors (list of floats)
    """

    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    if args.fingerprint_type == 'MPN': # only need to supply input features if using FFN latent representation and if model calls for them.
        validate_feature_sources = False
    else:
        validate_feature_sources = True
    update_prediction_args(predict_args=args, train_args=train_args, validate_feature_sources=validate_feature_sources)
    args: Union[FingerprintArgs, TrainArgs]

    #set explicit H option and reaction option
    reset_featurization_parameters()
    if args.atom_descriptors == 'feature':
        set_extra_atom_fdim(train_args.atom_features_size)

    if args.bond_descriptors == 'feature':
        set_extra_bond_fdim(train_args.bond_features_size)

    set_explicit_h(train_args.explicit_h)
    set_adding_hs(args.adding_h)
    set_keeping_atom_map(args.keeping_atom_map)
    if train_args.reaction:
        set_reaction(train_args.reaction, train_args.reaction_mode)
    elif train_args.reaction_solvent:
        set_reaction(True, train_args.reaction_mode)

    print('Loading data')
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator
        )
    else:
        full_data = get_data(path=args.test_path, smiles_columns=args.smiles_columns, target_columns=[], ignore_columns=[], skip_invalid_smiles=False,
                             args=args, store_row=True)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Set fingerprint size
    if args.fingerprint_type == 'MPN':
        if args.atom_descriptors == "descriptor": # special case when we have 'descriptor' extra dimensions need to be added
            total_fp_size = (args.hidden_size + test_data.atom_descriptors_size()) * args.number_of_molecules
        else:
            if args.reaction_solvent:
                total_fp_size = args.hidden_size + args.hidden_size_solvent
            else:
                total_fp_size = args.hidden_size * args.number_of_molecules
        if args.features_only:
            raise ValueError('With features_only models, there is no latent MPN representation. Use last_FFN fingerprint type instead.')
    elif args.fingerprint_type == 'last_FFN':
        if args.ffn_num_layers != 1:
            total_fp_size = args.ffn_hidden_size
        else:
            raise ValueError('With a ffn_num_layers of 1, there is no latent FFN representation. Use MPN fingerprint type instead.')
    else:
        raise ValueError(f'Fingerprint type {args.fingerprint_type} not supported')
    all_fingerprints = np.zeros((len(test_data), total_fp_size, len(args.checkpoint_paths)))

    # Load model
    print(f'Encoding smiles into a fingerprint vector from {len(args.checkpoint_paths)} models.')

    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths))):
        model = load_checkpoint(checkpoint_path, device=args.device)
        scaler, features_scaler, atom_descriptor_scaler, bond_descriptor_scaler, atom_bond_scaler = load_scalers(args.checkpoint_paths[index])

        # Normalize features
        if args.features_scaling or train_args.atom_descriptor_scaling or train_args.bond_descriptor_scaling:
            test_data.reset_features_and_targets()
            if args.features_scaling:
                test_data.normalize_features(features_scaler)
            if train_args.atom_descriptor_scaling and args.atom_descriptors is not None:
                test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
            if train_args.bond_descriptor_scaling and args.bond_descriptors is not None:
                test_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)

        # Make fingerprints
        model_fp = model_fingerprint(
            model=model,
            data_loader=test_data_loader,
            fingerprint_type=args.fingerprint_type
        )
        if args.fingerprint_type == 'MPN' and (args.features_path is not None or args.features_generator): # truncate any features from MPN fingerprint
            model_fp = np.array(model_fp)[:,:total_fp_size] 
        all_fingerprints[:,:,index] = model_fp

    # Save predictions
    print(f'Saving predictions to {args.preds_path}')
    # assert len(test_data) == len(all_fingerprints) #TODO: add unit test for this
    makedirs(args.preds_path, isfile=True)

    # Set column names
    fingerprint_columns = []
    if args.fingerprint_type == 'MPN':
        if len(args.checkpoint_paths) == 1:
            for j in range(total_fp_size//args.number_of_molecules):
                for k in range(args.number_of_molecules):
                    fingerprint_columns.append(f'fp_{j}_mol_{k}')
        else:
            for j in range(total_fp_size//args.number_of_molecules):
                for i in range(len(args.checkpoint_paths)):
                    for k in range(args.number_of_molecules):
                        fingerprint_columns.append(f'fp_{j}_mol_{k}_model_{i}')

    else: # args == 'last_FNN'
        if len(args.checkpoint_paths) == 1:
            for j in range(total_fp_size):
                fingerprint_columns.append(f'fp_{j}')
        else:
            for j in range(total_fp_size):
                for i in range(len(args.checkpoint_paths)):
                    fingerprint_columns.append(f'fp_{j}_model_{i}')

    # Copy predictions over to full_data
    for full_index, datapoint in enumerate(full_data):
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = all_fingerprints[valid_index].reshape((len(args.checkpoint_paths) * total_fp_size)) if valid_index is not None else ['Invalid SMILES'] * len(args.checkpoint_paths) * total_fp_size

        for i in range(len(fingerprint_columns)):
            datapoint.row[fingerprint_columns[i]] = preds[i]

    # Write predictions
    with open(args.preds_path, 'w', newline="") as f:
        writer = csv.DictWriter(f, fieldnames=args.smiles_columns+fingerprint_columns,extrasaction='ignore')
        writer.writeheader()
        for datapoint in full_data:
            writer.writerow(datapoint.row)

    if return_invalid_smiles:
        full_fingerprints = np.zeros((len(full_data), total_fp_size, len(args.checkpoint_paths)), dtype='object')
        for full_index in range(len(full_data)):
            valid_index = full_to_valid_indices.get(full_index, None)
            preds = all_fingerprints[valid_index] if valid_index is not None else np.full((total_fp_size, len(args.checkpoint_paths)), 'Invalid SMILES')
            full_fingerprints[full_index] = preds
        return full_fingerprints
    else:
        return all_fingerprints

def model_fingerprint(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            fingerprint_type: str = 'MPN',
            disable_progress_bar: bool = False) -> List[List[float]]:
    """
    Encodes the provided molecules into the latent fingerprint vectors, according to the provided model.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :return: A list of fingerprint vector lists.
    """
    model.eval()

    fingerprints = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_descriptors_batch, bond_features_batch = \
            batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_descriptors(), batch.bond_features()

        # Make predictions
        with torch.no_grad():
            batch_fp = model.fingerprint(mol_batch, features_batch, atom_descriptors_batch,
                                         atom_features_batch, bond_descriptors_batch,
                                         bond_features_batch, fingerprint_type)

        # Collect vectors
        batch_fp = batch_fp.data.cpu().tolist()

        fingerprints.extend(batch_fp)

    return fingerprints

def chemprop_fingerprint() -> None:
    """
    Parses Chemprop predicting arguments and returns the latent representation vectors for
    provided molecules, according to a previously trained model.
    """
    molecule_fingerprint(args=FingerprintArgs().parse_args())
