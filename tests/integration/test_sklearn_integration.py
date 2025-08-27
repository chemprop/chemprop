from sklearn.pipeline import Pipeline

from chemprop.sklearn_integration import ChempropMulticomponentTransformer, ChempropRegressor


def test_sklearn_pipeline(rxn_mol_regression_data):
    sklearnPipeline = Pipeline(
        [
            (
                "featurizer",
                ChempropMulticomponentTransformer(component_types=["reaction", "molecule"]),
            ),
            ("regressor", ChempropRegressor(epochs=50, val_size=0.1)),
        ]
    )
    rxns, smis, Y = rxn_mol_regression_data
    sklearnPipeline.fit(X=[rxns, smis], y=Y)
    score = sklearnPipeline.score(X=[rxns[:5], smis[:5]], y=Y[:5])
    assert score < 5
    sklearnPipeline["regressor"].save_model("checkpoints")
