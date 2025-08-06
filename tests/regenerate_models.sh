#!/bin/bash -l

CHEMPROP_ENV=$1
CHEMPROP_PATH=$2

if [ -z "${CHEMPROP_ENV}" ] || [ -z "${CHEMPROP_PATH}" ]; then
    echo "Usage: regenerate_models.sh <CHEMPROP_ENV> <CHEMPROP_PATH>"
    exit 1
fi

conda activate $CHEMPROP_ENV

# test_cli_classification_mol

rm -rf test_cli_classification_mol

chemprop train -i $CHEMPROP_PATH/tests/data/classification/mol.csv --accelerator cpu --epochs 3 --num-workers 0 --task-type classification --save-dir test_cli_classification_mol

cp -L test_cli_classification_mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_classification_mol.pt

# test_cli_classification_mol_multiclass

rm -rf test_cli_classification_mol_multiclass

chemprop train -i $CHEMPROP_PATH/tests/data/classification/mol_multiclass.csv --accelerator cpu --epochs 3 --num-workers 0 --save-dir test_cli_classification_mol_multiclass --task-type multiclass

cp -L test_cli_classification_mol_multiclass/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_classification_mol_multiclass.pt

# test_cli_regression_mol+mol

rm -rf test_cli_regression_mol+mol

chemprop train -i $CHEMPROP_PATH/tests/data/regression/mol+mol/mol+mol.csv --accelerator cpu --epochs 3 --num-workers 0 --smiles-columns smiles solvent --save-dir test_cli_regression_mol+mol

cp -L test_cli_regression_mol+mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol+mol.pt

cp -L test_cli_regression_mol+mol/model_0/checkpoints/best*.ckpt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol+mol.ckpt

# test_cli_regression_mol

rm -rf test_cli_regression_mol

chemprop train -i $CHEMPROP_PATH/tests/data/regression/mol/mol.csv --accelerator cpu --epochs 3 --num-workers 0 --save-dir test_cli_regression_mol

cp -L test_cli_regression_mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol.pt

cp -L test_cli_regression_mol/model_0/checkpoints/best*.ckpt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol.ckpt

# test_cli_regression_mol_multitask

rm -rf test_cli_regression_mol_multitask

chemprop train -i $CHEMPROP_PATH/tests/data/regression/mol_multitask.csv --accelerator cpu --epochs 3 --num-workers 0 --save-dir test_cli_regression_mol_multitask

cp -L test_cli_regression_mol_multitask/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol_multitask.pt

# test_cli_regression_rxn+mol

rm -rf test_cli_regression_rxn+mol

chemprop train -i $CHEMPROP_PATH/tests/data/regression/rxn+mol/rxn+mol.csv --accelerator cpu --epochs 3 --num-workers 0 --reaction-columns rxn_smiles --smiles-columns solvent_smiles --save-dir test_cli_regression_rxn+mol

cp -L test_cli_regression_rxn+mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_rxn+mol.pt

# test_cli_regression_rxn

rm -rf test_cli_regression_rxn

chemprop train -i $CHEMPROP_PATH/tests/data/regression/rxn/rxn.csv --accelerator cpu --epochs 3 --num-workers 0 --reaction-columns smiles --save-dir test_cli_regression_rxn

cp -L test_cli_regression_rxn/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_rxn.pt

cp -L test_cli_regression_rxn/model_0/checkpoints/best*.ckpt $CHEMPROP_PATH/tests/data/example_model_v2_regression_rxn.ckpt

# test_cli_regression_mve_mol

rm -rf test_cli_regression_mve_mol

chemprop train -i $CHEMPROP_PATH/tests/data/regression/mol/mol.csv --accelerator cpu --epochs 3 --num-workers 0 --save-dir test_cli_regression_mve_mol --task-type regression-mve

cp -L test_cli_regression_mve_mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mve_mol.pt

# test_cli_regression_evidential_mol

rm -rf test_cli_regression_evidential_mol

chemprop train -i $CHEMPROP_PATH/tests/data/regression/mol/mol.csv --accelerator cpu --epochs 3 --num-workers 0 --save-dir test_cli_regression_evidential_mol --task-type regression-evidential

cp -L test_cli_regression_evidential_mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_evidential_mol.pt

# test_cli_regression_quantile_mol

rm -rf test_cli_regression_quantile_mol

chemprop train -i $CHEMPROP_PATH/tests/data/regression/mol/mol.csv --accelerator cpu --epochs 3 --num-workers 0 --save-dir test_cli_regression_quantile_mol --task-type regression-quantile --alpha 0.1

cp -L test_cli_regression_quantile_mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_quantile_mol.pt

# test_cli_classification_dirichlet_mol

rm -rf test_cli_classification_dirichlet_mol

chemprop train -i $CHEMPROP_PATH/tests/data/classification/mol.csv --accelerator cpu --epochs 3 --num-workers 0 --save-dir test_cli_classification_dirichlet_mol --task-type classification-dirichlet

cp -L test_cli_classification_dirichlet_mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_classification_dirichlet_mol.pt

# test_cli_classification_dirichlet_mol

rm -rf test_cli_multiclass_dirichlet_mol

chemprop train -i $CHEMPROP_PATH/tests/data/classification/mol_multiclass.csv --accelerator cpu --epochs 3 --num-workers 0 --save-dir test_cli_multiclass_dirichlet_mol --task-type multiclass-dirichlet

cp -L test_cli_multiclass_dirichlet_mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_multiclass_dirichlet_mol.pt

# test_cli_regression_mol+mol+rxn_check_predictions

rm -rf test_cli_regression_mol+mol+rxn_check_predictions

chemprop train -i tests/data/regression/rxn+mol/rxn+mol.csv --accelerator cpu --epochs 3 --num-workers 0 --reaction-columns rxn_smiles --smiles-columns solvent_smiles solvent_smiles --atom-features-path 1 tests/data/regression/rxn+mol/atom_features.npz --no-atom-feature-scaling --atom-descriptors-path tests/data/regression/rxn+mol/atom_descriptors.npz --bond-features-path tests/data/regression/rxn+mol/bond_features.npz --descriptors-path tests/data/regression/rxn+mol/descriptors.npz --rxn-mode REAC_DIFF_BALANCE --multi-hot-atom-featurizer-mode RIGR --keep-h --molecule-featurizers morgan_count --save-dir test_cli_regression_mol+mol+rxn_check_predictions

cp -L test_cli_regression_mol+mol+rxn_check_predictions/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol+mol+rxn_check_predictions.pt

chemprop predict -i tests/data/regression/rxn+mol/rxn+mol.csv --accelerator cpu --num-workers 0 --reaction-columns rxn_smiles --smiles-columns solvent_smiles solvent_smiles --atom-features-path 1 tests/data/regression/rxn+mol/atom_features.npz --no-atom-feature-scaling --atom-descriptors-path tests/data/regression/rxn+mol/atom_descriptors.npz --bond-features-path tests/data/regression/rxn+mol/bond_features.npz --descriptors-path tests/data/regression/rxn+mol/descriptors.npz --rxn-mode REAC_DIFF_BALANCE --multi-hot-atom-featurizer-mode RIGR --keep-h --molecule-featurizers morgan_count --model-path $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol+mol+rxn_check_predictions.pt -o tests/data/data_for_test_preds_stay_same.pkl