from sklearn.base import RegressorMixin, BaseEstimator
import torch
import torch.nn as nn
import numpy as np
from chemprop.nn.predictors import RegressionFFN


class Regressor(RegressorMixin, BaseEstimator):
    def __init__(
        self, input_dim: int = 300, hidden_dim: int = 256, dropout: float = 0.0, device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains a simple feedforward regressor on the given data.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        self.model = RegressionFFN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=2,
            dropout=self.dropout,
            activation="relu",
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(100):  # small training loop
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X_tensor).squeeze()
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Outputs regression predictions from the FFN.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).squeeze().cpu().numpy()
        return preds
