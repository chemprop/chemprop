# Property Prediction
This repository contains message passing neural networks for molecular property prediction.

## Table of Contents

- [Installation](#installation)
- [Web Interface](#web-interface)
- [Data](#data)
- [Training](#training)
  * [Train/Validation/Test Splits](#train-validation-test-splits)
  * [Cross validation](#cross-validation)
  * [Ensembling](#ensembling)
  * [Hyperparameter Optimization](#hyperparameter-optimization)
- [Predicting](#predicting)
- [TensorBoard](#tensorboard)
- [Results](#results)

## Installation
Requirements:
 * cuda >= 8.0 + cuDNN
 * Python 3/conda: Please follow the installation guide on [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
   * Create a conda environment with `conda create -n <name> python=3.6`
   * Activate the environment with `conda activate <name>`
 * pytorch: Please follow the installation guide on [https://pytorch.org/](https://pytorch.org/)
   * Typically it's `conda install pytorch torchvision -c pytorch`
 * tensorflow: Needed for Tensorboard training visualization
   * CPU-only: `pip install tensorflow`
   * GPU: `pip install tensorflow-gpu`
 * RDKit: `conda install -c rdkit rdkit`
 * Other packages: `pip install -r requirements.txt`
 * Note that if you get warning messages about kyotocabinet, it's safe to ignore them.
   * This is a requirement from descriptastorus, but not necessary for the parts of descriptastorus that we use. 
   * See https://github.com/bp-kelley/descriptastorus for installation details if you want to install anyway.

Alternatively, you can use Docker. The provided Dockerfile, when built and run, should get you a container with a conda env
that has the proper packages installed for you to run the chemprop code. 
   
## Web Interface

For those less familiar with the command line, we also have a web interface which allows for basic training and predicting. To start the web interface (after doing the above installation), run `python web.py` and then go to [localhost:5000](localhost:5000) in a web browser.

![Training with our web interface](static/images/web_train.png "Train")

![Predicting with our web interface](static/images/web_predict.png "Predict")


## Data

The data file must be be a **CSV file with a header row**. For example:
```
smiles,NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53
CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,1,,,0,0,1,0,0,0,0
CCN1C(=O)NC(c2ccccc2)C1=O,0,0,0,0,0,0,0,,0,,0,0
...
```
Datasets from [MoleculeNet](http://moleculenet.ai/) and a 450K subset of the ChEMBL dataset from [http://www.bioinf.jku.at/research/lsc/index.html](http://www.bioinf.jku.at/research/lsc/index.html) have been preprocessed and are available in `data.tar.gz`. To uncompress them, run `tar xvzf data.tar.gz`.

## Training

To train a model, run:
```
python train.py --data_path <path> --dataset_type <type> --save_dir <dir>
```
where `<path>` is the path to a CSV file containing a dataset, `<type>` is either "classification" or "regression" depending on the type of the dataset, and `<dir>` is the directory where model checkpoints will be saved.

For example:
```
python train.py --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints
```

Notes:
* Classification is assumed to be binary.
* Empty values in the CSV are ignored.
* `--save_dir` may be left out if you don't want to save model checkpoints.
* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with `--metric <metric>`.
* `--quiet` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the `save_dir`.

### Train/Validation/Test Splits

Our code supports several ways of splitting data into train, validation, and test splits.

**Random:** By default, the data in `--data_path` will be split randomly into train, validation, and test sets.

**Scaffold:** Alternatively, the data can be split by molecular scaffold so that the same scaffold never appears in more than one split. This can be specified with `--split_type scaffold_balanced`.

**Separate val/test:** If you have a separate data file you would like to use as the validation or test set, you can specify it with `--separate_val_path <val_path>` and/or `--separate_test_path <test_path>`.

Note: Both random and scaffold  split the data into 80% train, 10% validation, and 10% test. Both also involve a random component and can be seeded with `--seed <seed>`. (A seed of 0 is used by default.)

### Cross validation

k-fold cross-validation can be run by specifying `--num_folds <k>` (which is 1 by default).

### Ensembling

To train an ensemble, specify the number of models in the ensemble with `--ensemble_size <n>` (which is 1 by default).

### Hyperparameter Optimization

Although the default message passing architecture works quite well on a variety of datasets, optimizing the hyperparameters for a particular dataset often leads to marked improvement in predictive performance. We have automated hyperparameter optimization via Bayesian optimization (using the [hyperopt](https://github.com/hyperopt/hyperopt) package) in `hyperparameter_optimization.py`. This script finds the optimal hidden size, depth, dropout, and number of feed-forward layers for our model. Optimization can be run as follows:
```
python hyperparameter_optimization.py --data_path <data_path> --dataset_type <type> --num_iters <n> --config_save_path <config_path>
```
where `num_iters` is the number of hyperparameter settings to try and `config_save_path` is the path to a `.json` file where the optimal hyperparameters will be saved. Once hyperparameter optimization is complete, the optimal hyperparameters can be applied during training by specifying the `config_path` as follows:
```
python train.py --data_path <data_path> --dataset_type <type> --config_path <config_path>
```
 
## Predicting

To load a trained model and make predictions, run `predict.py` and specify:
* `--test_path <path>` Path to the data to predict on.
* A checkpoint by using either:
  * `--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. `--save_dir` during training). This will walk the directory, load all `.pt` files it finds, and treat the models as an ensemble.
  * `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
* `--preds_path` Path where a CSV file containing the predictions will be saved.

For example:
```
python predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints --preds_path tox21_preds.csv
```
or
```
python predict.py --test_path data/tox21.csv --checkpoint_path tox21_checkpoints/fold_0/model_0/model.pt --preds_path tox21_preds.csv
```

## TensorBoard

During training, TensorBoard logs are automatically saved to the same directory as the model checkpoints. To view TensorBoard logs, run `tensorboard --logdir=<dir>` where `<dir>` is the path to the checkpoint directory. Then navigate to [http://localhost:6006](http://localhost:6006).

## Results

**NOTE:** These results are out of date and will be updated shortly.

We compared our model against the graph convolution in deepchem. Our results are averaged over 3 runs with different random seeds, namely different splits across datasets. Unless otherwise indicated, all models were trained using hidden size 1800, depth 6, and master node. We did a few hyperparameter experiments on qm9, but did no searching on the other datasets, so there may still be further room for improvement.

Results on classification datasets (AUC score, the higher the better)

| Dataset | Size |	Ours |	GraphConv (deepchem) |
| :---: | :---: | :---: | :---: |
| Bace | 1,513 | 0.884 ± 0.034	| 0.783 ± 0.014 |
| BBBP | 2,039 | 0.922 ± 0.012	| 0.690 ± 0.009 |
| Tox21 | 7,831 | 0.851 ± 0.015	| 0.829 ± 0.006 |
| Toxcast | 8,576 | 0.748 ± 0.014	| 0.716 ± 0.014 |
| Sider | 1,427 |	0.643 ± 0.027	| 0.638 ± 0.012 |
| clintox | 1,478 | 0.882 ± 0.022	| 0.807 ± 0.047 |
| MUV | 93,087 | 0.067 ± 0.03* | 0.046 ± 0.031 |
| HIV | 41,127 |	0.821 ± 0.034† |	0.763 ± 0.016 |
| PCBA | 437,928 | 0.218 ± 0.001* | 	0.136 ± 0.003 | 

Results on regression datasets (score, the lower the better)

Dataset | Size | Ours | GraphConv/MPNN (deepchem) |
| :---: | :---: | :---: | :---: |
delaney	| 1,128 | 0.687 ± 0.037 | 	0.58 ± 0.03 |
Freesolv | 642 |	0.915 ± 0.154	| 1.15 ± 0.12 |
Lipo | 4,200 |	0.565 ± 0.052 |	0.655 ± 0.036 |
qm8 | 21,786 |	0.008 ± 0.000 | 0.0143 ± 0.0011 |
qm9 | 133,884 |	2.47 ± 0.036	| 3.2 ± 1.5 |

†HIV was trained with hidden size 1800 and depth 6 but without the master node.
*MUV and PCBA are using a much older version of the model.

Lastly, you can find the code to our original repo at https://github.com/wengong-jin/chemprop and for the Mayr et al. baseline at https://github.com/yangkevin2/lsc_experiments . 
