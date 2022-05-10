from typing import List
import numpy as np

from chemprop.data import MoleculeDataset


def get_reshaped_values(
    values: List[List[List[float]]],
    test_data: MoleculeDataset,
    atom_targets: List[str],
    bond_targets: List[str],
    num_tasks: int,
) -> List[List[List[float]]]:
    """
    Reshape the input from shape (num_tasks, numer of atomic/bond properties for each task, 1)
    to shape (data_size, num_tasks, number of atomic/bond properties for this data in each task).

    :param values: List of atomic/bond properteis with shape
                   (num_tasks, numer of atomic/bond properties for each task, 1).
    :param test_data: A :class:`~chemprop.data.MoleculeDataset` containing valid datapoints.
    :param atom_targets: List of the names of target atomic values.
    :param bond_targets: List of the names of target bond values.
    :param num_tasks: Number of tasks.
    :return: List of atomic/bond properteis with shape
             (data_size, num_tasks, number of atomic/bond properties for this data in each task).
    """
    n_atoms, n_bonds = test_data.number_of_atoms, test_data.number_of_bonds
    reshaped_values = []
    for i, atom_target in enumerate(atom_targets):
        reshaped_values.append(
            np.split(values[i].flatten(), np.cumsum(np.array(n_atoms)))[:-1]
        )
    for i, bond_target in enumerate(bond_targets):
        reshaped_values.append(
            np.split(
                values[i + len(atom_targets)].flatten(), np.cumsum(np.array(n_bonds))
            )[:-1]
        )
    reshaped_values = [
        [reshaped_values[j][i] for j in range(num_tasks)] for i in range(len(test_data))
    ]
    return reshaped_values


def get_reshaped_individual_preds(
    individual_preds: List[List[List[List[float]]]],
    test_data: MoleculeDataset,
    atom_targets: List[str],
    bond_targets: List[str],
    num_tasks: int,
    num_models: int,
) -> List[List[List[List[float]]]]:
    """
    Reshape the input from shape (num_tasks, numer of atomic/bond properties for each task, 1, num_models)
    to shape (data_size, num_tasks, num_models, number of atomic/bond properties for this data in each task).

    :param individual_preds: List of atomic/bond properteis with shape
                             (num_tasks, numer of atomic/bond properties for each task, 1, num_models).
    :param test_data: A :class:`~chemprop.data.MoleculeDataset` containing valid datapoints.
    :param atom_targets: List of the names of target atomic values.
    :param bond_targets: List of the names of target bond values.
    :param num_tasks: Number of tasks.
    :param num_models: Number of models.
    :return: List of atomic/bond properteis with shape
             (data_size, num_tasks, num_models, number of atomic/bond properties for this data in each task).
    """
    n_atoms, n_bonds = test_data.number_of_atoms, test_data.number_of_bonds
    individual_values = [[[] for j in range(num_tasks)] for i in range(len(test_data))]
    for i, atom_target in enumerate(atom_targets):
        for j in range(num_models):
            atom_target = np.split(
                individual_preds[i][:, :, j].flatten(), np.cumsum(np.array(n_atoms))
            )[:-1]
            for k, target in enumerate(atom_target):
                individual_values[k][i].append(target)
    for i, bond_target in enumerate(bond_targets):
        for j in range(num_models):
            bond_target = np.split(
                individual_preds[i + len(atom_targets)][:, :, j].flatten(),
                np.cumsum(np.array(n_bonds)),
            )[:-1]
            for k, target in enumerate(bond_target):
                individual_values[k][i + len(atom_targets)].append(target)
    return individual_values
