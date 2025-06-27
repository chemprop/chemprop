from typing import Optional, Union
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.molgraph.reaction import CondensedGraphOfReactionFeaturizer, RxnMode
from chemprop.models.model import MPNN
from transformer import build_default_mpnn


class ChempropTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, featurizer_type="molecule", rxn_mode=RxnMode.REAC_DIFF):
        self.rxn_mode = rxn_mode
        if featurizer_type == "molecule":
            self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        elif featurizer_type == "reaction":
            self.featurizer = CondensedGraphOfReactionFeaturizer(mode_=rxn_mode)
        else:
            raise ValueError("featurizer_type must be 'molecule' or 'reaction'")
        self.featurizer_type = featurizer_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._to_molgraph(mol) for mol in X]

    def _to_molgraph(self, mol: Union[str, tuple[str, str]]) -> MolGraph:
        if self.featurizer_type == "molecule":
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
            return self.featurizer(mol)
        elif self.featurizer_type == "reaction":
            if isinstance(mol, tuple):
                reac, prod = mol
                reac = Chem.MolFromSmiles(reac)
                prod = Chem.MolFromSmiles(prod)
                return self.featurizer((reac, prod))
            else:
                raise ValueError("Reaction inputs must be tuples of SMILES strings")


class ChempropRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, mpnn_args: Optional[dict] = None, device="cpu", n_epochs=100, lr=1e-3):
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.mpnn_args = mpnn_args
        self.model = None

    def fit(self, X, y):
        if self.model is None:
            input_dim = X[0].V.shape[1]
            edge_dim = X[0].E.shape[1]
            if self.mpnn_args is None:
                self.model = build_default_mpnn(input_dim, edge_dim)
            else:
                self.model = MPNN(**self.mpnn_args)

        self.model.train()
        batch = BatchMolGraph(mgs=X)
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
        batch = BatchMolGraph(mgs=X)
        with torch.no_grad():
            preds = self.model(batch)
        return preds.view(-1).cpu().numpy()

    def load_from_file(self, checkpoint_path: str, strict: bool = True):
        self.model = MPNN.load_from_file(checkpoint_path, map_location=self.device, strict=strict)
        self.model.eval()
        return self


# microtest
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import cross_val_score

# sklearnPipeline = Pipeline([
#     ("featurizer", ChempropTransformer()),
#     ("regressor", ChempropRegressor().load_from_file("./tests/data/example_model_v2_classification_dirichlet_mol.pt"))
# ])

# X = np.array([
#     "CCO", "CCN", "CCC", "COC", "CNC", "CCCl", "CCBr", "CCF", "CCI", "CC=O",
#     "CC#N", "CC(C)O", "CC(C)N", "CC(C)C", "COC(C)", "CN(C)C", "C1CCCCC1", "C1=CC=CC=C1",
#     "CC(C)(C)O", "CC(C)(C)N", "COCCO", "CCOC(=O)C", "CCN(CC)CC", "CN1CCCC1", "C(CO)N"
# ])

# y = np.array([
#     0.50, 0.60, 0.55, 0.58, 0.52, 0.62, 0.65, 0.57, 0.59, 0.61,
#     0.56, 0.60, 0.54, 0.53, 0.62, 0.63, 0.45, 0.40,
#     0.64, 0.66, 0.59, 0.51, 0.48, 0.46, 0.49
# ])

# scores = cross_val_score(sklearnPipeline, X, y, cv=5, scoring='neg_mean_squared_error')
# print("Cross-validation scores:", scores)
# print("Mean MSE:", -scores.mean())


# #microtest
# sklearnPipeline1 = Pipeline([
#     ("featurizer", ChempropTransformer()),
#     ("regressor", ChempropRegressor())
# ])

# X_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
# y_targets = [0.5, 1.2, 0.7]
# sklearnPipeline1.fit(X_smiles, y_targets)

# X_test = ["CCN", "CCCl"]
# predictions = sklearnPipeline1.predict(X_test)
# print("Predictions:", predictions)

# import joblib
# joblib.dump(sklearnPipeline1, "chemprop_pipeline.pkl")

# loadedPipeline = joblib.load("chemprop_pipeline.pkl")
# print("Reproduced Predictions:", loadedPipeline.predict(X))
