from typing import List
import numpy as np

from chemprop.data import MoleculeDataset


def reshape_values(
    values: List[List[List[float]]],
    test_data: MoleculeDataset,
    natom_targets: int,
    nbond_targets: int,
) -> List[List[List[float]]]:
    """
    Reshape the input from shape (num_tasks, number of atomic/bond properties for each task, 1)
    to shape (data_size, num_tasks, number of atomic/bond properties for this data in each task).

    :param values: List of atomic/bond properties with shape
                   (num_tasks, number of atomic/bond properties for each task, 1).
    :param test_data: A :class:`~chemprop.data.MoleculeDataset` containing valid datapoints.
    :param natom_targets: The number of atomic targets.
    :param nbond_targets: The number of bond targets.
    :return: List of atomic/bond properties with shape
             (data_size, num_tasks, number of atomic/bond properties for this data in each task).
    """
    n_atoms, n_bonds = test_data.number_of_atoms, test_data.number_of_bonds
    num_atom_bond_tasks = natom_targets + nbond_targets
    reshaped_values = np.empty([len(test_data), num_atom_bond_tasks], dtype=object)

    for i in range(natom_targets):
        atom_targets = values[i].reshape(-1,)
        atom_targets = np.hsplit(atom_targets, np.cumsum(np.array(n_atoms)))[:-1]
        reshaped_values[:, i] = atom_targets

    for i in range(nbond_targets):
        bond_targets = values[i+ natom_targets].reshape(-1,)
        bond_targets = np.hsplit(bond_targets, np.cumsum(np.array(n_bonds)))[:-1]
        reshaped_values[:, i + natom_targets] = bond_targets

    return reshaped_values


def reshape_individual_preds(
    individual_preds: List[List[List[List[float]]]],
    test_data: MoleculeDataset,
    natom_targets: int,
    nbond_targets: int,
    num_models: int,
) -> List[List[List[List[float]]]]:
    """
    Reshape the input from shape (num_tasks, number of atomic/bond properties for each task, 1, num_models)
    to shape (data_size, num_tasks, num_models, number of atomic/bond properties for this data in each task).

    :param individual_preds: List of atomic/bond properties with shape
                             (num_tasks, number of atomic/bond properties for each task, 1, num_models).
    :param test_data: A :class:`~chemprop.data.MoleculeDataset` containing valid datapoints.
    :param natom_targets: The number of atomic targets.
    :param nbond_targets: The number of bond targets.
    :param num_models: Number of models.
    :return: List of atomic/bond properties with shape
             (data_size, num_tasks, num_models, number of atomic/bond properties for this data in each task).
    """
    n_atoms, n_bonds = test_data.number_of_atoms, test_data.number_of_bonds
    num_atom_bond_tasks = natom_targets + nbond_targets
    individual_values = np.empty([len(test_data), num_atom_bond_tasks], dtype=object)

    for i in range(natom_targets):
        atom_targets = individual_preds[i].T.reshape(num_models, -1)
        atom_targets = np.hsplit(atom_targets, np.cumsum(np.array(n_atoms)))[:-1]
        individual_values[:, i] = atom_targets

    for i in range(nbond_targets):
        bond_targets = individual_preds[i + natom_targets].T.reshape(num_models, -1)
        bond_targets = np.hsplit(bond_targets, np.cumsum(np.array(n_bonds)))[:-1]
        individual_values[:, i + natom_targets] = bond_targets

    return individual_values
