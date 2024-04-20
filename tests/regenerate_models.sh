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

chemprop train -i $CHEMPROP_PATH/tests/data/classification/mol.csv --accelerator cpu --epochs 1 --num-workers 0 --task-type classification --save-dir test_cli_classification_mol

cp -L test_cli_classification_mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_classification_mol.pt

# test_cli_classification_mol_multiclass

rm -rf test_cli_classification_mol_multiclass

chemprop train -i $CHEMPROP_PATH/tests/data/classification/mol_multiclass.csv --accelerator cpu --epochs 1 --num-workers 0 --save-dir test_cli_classification_mol_multiclass --task-type multiclass

cp -L test_cli_classification_mol_multiclass/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_classification_mol_multiclass.pt

# test_cli_regression_mol+mol

rm -rf test_cli_regression_mol+mol

chemprop train -i $CHEMPROP_PATH/tests/data/regression/mol+mol/mol+mol.csv --accelerator cpu --epochs 1 --num-workers 0 --smiles-columns smiles solvent --save-dir test_cli_regression_mol+mol

cp -L test_cli_regression_mol+mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol+mol.pt

cp -L test_cli_regression_mol+mol/model_0/checkpoints/best*.ckpt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol+mol.ckpt

# test_cli_regression_mol

rm -rf test_cli_regression_mol

chemprop train -i $CHEMPROP_PATH/tests/data/regression/mol/mol.csv --accelerator cpu --epochs 1 --num-workers 0 --save-dir test_cli_regression_mol

cp -L test_cli_regression_mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol.pt

cp -L test_cli_regression_mol/model_0/checkpoints/best*.ckpt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol.ckpt

# test_cli_regression_mol_multitask

rm -rf test_cli_regression_mol_multitask

chemprop train -i $CHEMPROP_PATH/tests/data/regression/mol_multitask.csv --accelerator cpu --epochs 1 --num-workers 0 --save-dir test_cli_regression_mol_multitask

cp -L test_cli_regression_mol_multitask/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_mol_multitask.pt

# test_cli_regression_rxn+mol

rm -rf test_cli_regression_rxn+mol

chemprop train -i $CHEMPROP_PATH/tests/data/regression/rxn+mol/rxn+mol.csv --accelerator cpu --epochs 1 --num-workers 0 --reaction-columns rxn_smiles --smiles-columns solvent_smiles --save-dir test_cli_regression_rxn+mol

cp -L test_cli_regression_rxn+mol/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_rxn+mol.pt

# test_cli_regression_rxn

rm -rf test_cli_regression_rxn

chemprop train -i $CHEMPROP_PATH/tests/data/regression/rxn/rxn.csv --accelerator cpu --epochs 1 --num-workers 0 --reaction-columns smiles --save-dir test_cli_regression_rxn

cp -L test_cli_regression_rxn/model_0/best.pt $CHEMPROP_PATH/tests/data/example_model_v2_regression_rxn.pt

cp -L test_cli_regression_rxn/model_0/checkpoints/best*.ckpt $CHEMPROP_PATH/tests/data/example_model_v2_regression_rxn.ckpt
