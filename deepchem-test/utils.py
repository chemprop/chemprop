import random
from typing import List, Tuple


def get_data(path: str) -> List[Tuple[str, List[float]]]:
    """
    Gets smiles string and target values from a CSV file.

    :param path: Path to a CSV file.
    :return: A list of tuples where each tuple contains a smiles string and
    a list of target values (which are None if the target value is not specified).
    """
    data = []
    with open(path) as f:
        f.readline()  # remove header

        for line in f:
            line = line.strip().split(',')
            smiles = line[0]
            values = [float(x) if x != '' else None for x in line[1:]]
            data.append((smiles, values))

    return data


def split_data(data: List[Tuple[str, List[float]]],
               sizes: Tuple[float] = (0.8, 0.1, 0.1),
               seed: int = 0) -> Tuple[List[Tuple[str, List[float]]],
                                       List[Tuple[str, List[float]]],
                                       List[Tuple[str, List[float]]]]:
    """
    Splits data into training, validation, and test splits.

    :param data: A list of data points (smiles string, target values).
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3, sum(sizes) == 1

    random.seed(seed)
    random.shuffle(data)

    train_size, val_size = [int(size * len(data)) for size in sizes[:2]]

    train = data[:train_size]
    val = data[train_size:train_size + val_size]
    test = data[train_size + val_size:]

    return train, val, test
