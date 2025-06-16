from sklearn.base import TransformerMixin, BaseEstimator
from typing import List, Union, Optional
from rdkit import Chem
import numpy as np
import torch

from chemprop.models.model import MPNN
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from chemprop.data.molgraph import MolGraph
from chemprop.nn.message_passing import AtomMessagePassing
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.predictors import RegressionFFN
from chemprop.data.collate import BatchMolGraph


def build_default_mpnn(
    atom_input_dim: int,
    bond_input_dim: int,
    hidden_dim: int = 300,
    depth: int = 3,
    dropout: float = 0.0,
) -> MPNN:
    message_passing = AtomMessagePassing(
        d_v=atom_input_dim,
        d_e=bond_input_dim,
        d_h=hidden_dim,
        depth=depth,
        dropout=dropout,
        activation="relu",
        bias=True
    )

    agg = MeanAggregation()

    predictor = RegressionFFN(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        n_layers=2,
        dropout=dropout,
        activation="relu",
    )

    return MPNN(
        message_passing=message_passing,
        agg=agg,
        predictor=predictor,
        batch_norm=True,
        metrics=[],
    )

class MPNNTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, mpnn_args: Optional[dict] = None, device: Optional[str] = "cpu"):
        self.mpnn_args = mpnn_args
        self.device = device
        self.model = None
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()

    def fit(self, X: List[Union[str, Chem.Mol]], y=None):
        """
        Initializes the MPNN model for embedding extraction.
        """
        if self.mpnn_args is not None:
            self.model = MPNN(**self.mpnn_args).to(self.device)
        else:
            self.model = build_default_mpnn(self.featurizer.atom_fdim, 
                                            self.featurizer.bond_fdim,
                                            hidden_dim=300,
                                            depth=3).to(self.device)
        self.model.eval()
        return self

    def _to_molgraph(self, mol: Union[str, Chem.Mol]) -> MolGraph:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        return self.featurizer(mol)

    def transform(self, X: List[Union[str, Chem.Mol]]) -> np.ndarray:
        """
        Converts a list of SMILES or Mol objects to molecular embeddings.
        """
        mol_graphs = [self._to_molgraph(mol) for mol in X]

        # The MPNN model expects batched MolGraphs
        batch = BatchMolGraph(mgs = mol_graphs)
        batch.to(self.device)
        with torch.no_grad():
            embeddings = self.model.fingerprint(batch)

        return embeddings.cpu().numpy()