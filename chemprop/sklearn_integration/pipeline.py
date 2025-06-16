from sklearn.pipeline import Pipeline
from chemprop.sklearn_integration.transformer import MPNNTransformer
from chemprop.sklearn_integration.regressor import Regressor

mpnn_pipeline = Pipeline([
    ("embedding", MPNNTransformer()),
    ("regressor", Regressor(input_dim=300))
])

X_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
y_targets = [0.5, 1.2, 0.7]

mpnn_pipeline.fit(X_smiles, y_targets)

X_test = ["CCN", "CCCl"]
predictions = mpnn_pipeline.predict(X_test)

print("Predictions:", predictions)