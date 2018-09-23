from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class ScaffoldGenerator:
    """
    Generate molecular scaffolds.
    Parameters
    ----------
    include_chirality : : bool, optional (default False)
        Include chirality in scaffolds.
    """

    def __init__(self, include_chirality: bool = False):
        self.include_chirality = include_chirality

    def get_scaffold(self, mol: Chem.rdchem.Mol) -> str:
        """
        Get Murcko scaffolds for molecules.
        Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
        essentially that part of the molecule consisting of rings and the
        linker atoms between them.
        Parameters
        ----------
        mol : array_like
            Molecules.
        """
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=self.include_chirality)


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    engine = ScaffoldGenerator(include_chirality=include_chirality)
    scaffold = engine.get_scaffold(mol)
    return scaffold


def scaffold_split(dataset, frac_train: float = .8, frac_valid: float = .1, frac_test: float = .1):
    scaffolds = {}
    data_len = len(dataset)
    for ind, tup in enumerate(dataset):
        smiles = tup[0]
        scaffold = generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_inds, valid_inds, test_inds = [], [], []

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set

    train = [dataset[i] for i in train_inds]
    valid = [dataset[i] for i in valid_inds]
    test = [dataset[i] for i in test_inds]
    return train, valid, test
