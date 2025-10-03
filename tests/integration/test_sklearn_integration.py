from sklearn.pipeline import Pipeline

from chemprop.sklearn_integration import ChempropMulticomponentTransformer, ChempropRegressor


def test_sklearn_pipeline(rxn_mol_regression_data):
    sklearnPipeline = Pipeline(
        [
            (
                "featurizer",
                ChempropMulticomponentTransformer(component_types=["molecule", "reaction"]),
            ),
            ("regressor", ChempropRegressor(epochs=100)),
        ]
    )
    rxns, smis, Y = rxn_mol_regression_data
    Y = Y.flatten()
    sklearnPipeline.fit(X=[smis, rxns], y=Y)
    pred = sklearnPipeline.predict(X=[smis, rxns])
    score = sklearnPipeline.score(X=[smis, rxns], y=Y)
    assert score < 1
    sklearnPipeline["regressor"].save_model("checkpoints")
