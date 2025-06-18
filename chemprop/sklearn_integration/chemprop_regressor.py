import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score

from typing import Optional, Union
from rdkit import Chem
from chemprop.data.molgraph import MolGraph
from chemprop.models.model import MPNN
from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer

from transformer import build_default_mpnn

class ChempropRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, mpnn_args: Optional[dict] = None, device="cpu", n_epochs=100, lr=1e-3):
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.mpnn_args = mpnn_args
        if mpnn_args is None:
            self.model = build_default_mpnn(self.featurizer.atom_fdim, self.featurizer.bond_fdim)
        else:
            self.model = MPNN(**mpnn_args)


    def fit(self, X, y):
        self.model.train()
        mol_graphs = [self._to_molgraph(mol) for mol in X]
        batch = BatchMolGraph(mgs = mol_graphs)

        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = self.model.criterion

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            preds = self.model(batch)
            loss = loss_fn(preds, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        mol_graphs = [self._to_molgraph(mol) for mol in X]
        batch = BatchMolGraph(mgs = mol_graphs)

        with torch.no_grad():
            preds = self.model(batch)

        return preds.view(-1).cpu().numpy()

    def _to_molgraph(self, mol: Union[str, Chem.Mol]) -> MolGraph:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        return self.featurizer(mol)


X = np.array([
    "CCO", "CCN", "CCC", "COC", "CNC", "CCCl", "CCBr", "CCF", "CCI", "CC=O",
    "CC#N", "CC(C)O", "CC(C)N", "CC(C)C", "COC(C)", "CN(C)C", "C1CCCCC1", "C1=CC=CC=C1",
    "CC(C)(C)O", "CC(C)(C)N", "COCCO", "CCOC(=O)C", "CCN(CC)CC", "CN1CCCC1", "C(CO)N"
])

y = np.array([
    0.50, 0.60, 0.55, 0.58, 0.52, 0.62, 0.65, 0.57, 0.59, 0.61,
    0.56, 0.60, 0.54, 0.53, 0.62, 0.63, 0.45, 0.40,
    0.64, 0.66, 0.59, 0.51, 0.48, 0.46, 0.49
])

from transformer import build_default_mpnn
scores = cross_val_score(ChempropRegressor(), X, y, cv=5, scoring='neg_mean_squared_error')
print(scores)