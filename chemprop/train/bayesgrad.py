from typing import List, Tuple, Dict, Union
from chemprop.args import PredictArgs, TrainArgs

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.features import mol2graph


def bayes_predict(model: nn.Module,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None,
            num_tasks: int = None,
            number_of_molecules: int = None) -> Tuple[
                List[List[float]],
                Dict[int, Dict[int, Dict[str, Dict[int, float]]]]
            ]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data_loader: A MoleculeDataLoader.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A StandardScaler object fit on the training targets.
    :param num_tasks: How may tasks are we predicting
    :param number_of_molecules: How many input molecules are there
    :return: A tuple where:
        entry 0: A list of lists of predictions. The outer list is examples while the inner list is tasks.
        entry 1: A dict mapping target# to:
                    dictionary mapping molecule# to dictionary with entries:
                        'atoms': dict mapping atom id to gradient sum
                        'bonds': dict mapping bond id to gradient sum
    """
    model.eval()

    preds = []

    # Return aggregated gradients for each target we are predicting
    # First level of dict is task
    # Second level of dict is smiles (in case of multiple smiles)
    grads = {i: {j: [] for j in range(number_of_molecules)} for i in range(num_tasks)}

    for batch in tqdm(data_loader, disable=disable_progress_bar):

        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, atom_descriptors_batch = batch.batch_graph(), batch.features(), batch.atom_descriptors()

        # For all :class:`MoleculeDatapoint` enable gradients
        for i in range(len(mol_batch)):

            # We wish to retain the gradients of the f_bonds
            mol_batch[i].f_bonds.requires_grad = True

        # Run predictions
        batch_preds = model(mol_batch, features_batch, atom_descriptors_batch)

        # How many of the features on each feature-column belong to atoms?
        n_atom_features = mol_batch[0].f_atoms.shape[1]

        # Get gradients for each output target
        for target_id in range(batch_preds.shape[1]):

            # Zero out all gradients in the model
            model.zero_grad()

            # The target gradients we wish to back-propagate are all zeros, except for the target
            # property for which we want to find the gradients with respect to each input
            target = torch.zeros_like(batch_preds)
            target[:, target_id] = 1

            # Run back propagation of gradients. Retain the graph afterwards, since we may need
            # it for subsequent targets
            batch_preds.backward(target, retain_graph=True)

            # Go through each molecule (in case of multiple)
            for i in range(number_of_molecules):

                # Get the sum of the gradients for atoms and bonds; we take the sum to get the overall way in which changes to a given input influences the target output
                #
                # EXPLANATION: f_bonds contains one entry for each bond-atom set in all the molecules in the batch, see featurization.py l179-195 for details,
                # where the first 0:n_atoms features belong to an atom, and the last n_atoms: belong to a bond
                #
                # In addition, we want a signed importance score, and as such we follow Eq. 6 in the paper
                # "BayesGrad: Explaining Predictions of Graph Convolutional Networks"
                # where the gradient is multiplied with the input features. Since most of our features are one-hot-encoding,
                # the added benefit is that we only look at the sparse entries in the input actually doing something.
                signed_gradient = mol_batch[i].f_bonds.grad * mol_batch[i].f_bonds
                atom_gradients = signed_gradient[:, 0:n_atom_features].sum(axis=1)
                bond_gradients = signed_gradient[:, n_atom_features:].sum(axis=1)

                # Get the sum of gradients on each atom and bond in each molecule
                #
                # EXPLANATION: Currently all atoms/bonds gradients for the entire minibatch are contained in the atom_gradients and bond_gradients tensors
                # We wish to get the sum of gradients on each bond and atom in each SMILES string in the minibatch. To do this we need to use the mappings
                # found in the BatchMolGraph object to get the features pertaining to each molecule. Specifically, the BatchMolGraph object contains:
                #
                # - b_scope: list of (start, n_bonds) tuples indicating which tensor features belong to which molecule
                # - a_scope: list of (_, n_atoms) tuples indicating for each smiles, how many atoms it contains
                # - b2a: list of int indicating for bond feature, which atom index has been used to add atom features.
                #
                # The ordering of atoms / bonds indexes within each molecule is specifically set up to match the loop
                # defined in the `chemprop.features.featurization.MolGraph` class.
                #
                # The atom index in batch.b2a is always the atom index of the molecule + the number of all atoms in the batch
                # which came before. We start the counter at 1 instead of 0, because the first entry in b2a is zero-padding.
                atom_count = 1

                # Store the aggregated gradient on each atom/bond for each SMILES in the batch in this list
                batch_grad_agg = []

                # Loop through all the molecule scopes, and collect the bond/atom gradient sums
                for ((start, n_bonds), (_, n_atoms)) in zip(mol_batch[i].b_scope, mol_batch[i].a_scope):

                    # Store molecule information in this dictionary
                    # 'atoms' dict maps atom indexes to gradient sum on that atom
                    # 'bonds' dict maps bond indexes to gradient sum on that bond
                    molecule_grad_agg = {'atoms': {}, 'bonds': {}}

                    # Go through each bond-atom feature.
                    for bond_idx, feature_idx in enumerate(np.arange(start, start+n_bonds)):

                        # Get atom index in molecule. We have to account for number of total atoms iterated so far
                        atom_idx = mol_batch[i].b2a[feature_idx].item() - atom_count

                        # Store the data for molecule: atom idx's can occur multiple times,
                        # so add to a list that will be summed in the end
                        molecule_grad_agg['atoms'].setdefault(atom_idx, []).append(atom_gradients[feature_idx].item())

                        # Store the data for molecule: bond idx's occur only once.
                        # They are directional though, so have to be summed during visualization!
                        molecule_grad_agg['bonds'][bond_idx] = bond_gradients[feature_idx].item()

                    # Atoms can have contributions from multiple bonds, so sum all those up
                    molecule_grad_agg['atoms'] = {k: sum(v) for k, v in molecule_grad_agg['atoms'].items()}

                    # mol_contributions now has all the contributions for a single SMILES string for single target. Add it to the minibatch
                    batch_grad_agg.append(molecule_grad_agg)

                    # Increase the atom count in order to go to next molecule
                    atom_count += n_atoms

                # Zero all the gradients on the input batch, so that we can collect gradients for next target
                mol_batch[i].f_bonds.grad.zero_()

                # Add the interpretations for this target to the final return result
                grads[target_id][i].extend(batch_grad_agg)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds, grads
