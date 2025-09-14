from sklearn.pipeline import Pipeline

from chemprop.sklearn_integration import ChempropMulticomponentTransformer, ChempropRegressor


def test_sklearn_pipeline(rxn_mol_regression_data):
    sklearnPipeline = Pipeline(
        [
            (
                "featurizer",
                ChempropMulticomponentTransformer(component_types=["reaction", "molecule"]),
            ),
            ("regressor", ChempropRegressor(epochs=200)),
        ]
    )
    rxns, smis, Y = rxn_mol_regression_data
    sklearnPipeline.fit(X=[rxns, smis], y=Y)
    score = sklearnPipeline.score(X=[rxns, smis], y=Y)
    assert score < 0.5
    sklearnPipeline["regressor"].save_model("checkpoints")
