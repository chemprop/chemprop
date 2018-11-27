# Property Prediction
This repository contains graph convolutional networks (or message passing network) for molecule property prediction.

## Table of Contents

* [Installation](#installation)
* [Data](#data)
* [Training](#training)
  + [Train/Validation/Test Split](#train-validation-test-split)
    - [Random split](#random-split)
    - [Separate test set](#separate-test-set)
  + [Cross validation](#cross-validation)
  + [Ensembling](#ensembling)
  + [Model hyperparameters and augmentations](#model-hyperparameters-and-augmentations)
* [Predicting](#predicting)
* [TensorBoard](#tensorboard)
* [Deepchem test](#deepchem-test)
  + [Results](#results)

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

## Data

The data file must be be a **CSV file with a header row**. For example:
```
smiles,NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53
CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,1,,,0,0,1,0,0,0,0
CCN1C(=O)NC(c2ccccc2)C1=O,0,0,0,0,0,0,0,,0,,0,0
...
```
Data sets from [deepchem](http://moleculenet.ai/) are available in the `data` directory.

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
* The default metric for classification is AUC and the default metric for regression is RMSE. The qm8 and qm9 datasets use MAE instead of RMSE, so you need to specify `--metric mae`.

### Train/Validation/Test Split

#### Random split

By default, the data in `--data_path` will be split randomly into train, validation, and test sets using the seed specified by `--seed` (default = 0). By default, the train set contains 80% of the data while the validation and test sets contain 10% of the data each. These sizes can be controlled with `--split_sizes` (for example, the default would be `--split_sizes 0.8 0.1 0.1`).

#### Separate test set

To use a different data set for testing, specify `--separate_test_path`. In this case, the data in `--data_path` will be split into only train and validation sets (80% and 20% of the data), and the test set will contain all the dta in `--separate_test_path`.

### Cross validation

k-fold cross-validation can be run by specifying the `--num_folds` argument (which is 1 by default). For example:
```
python train.py --data_path data/tox21.csv --dataset_type classification --num_folds 5
```

### Ensembling

To train an ensemble, specify the number of models in the ensemble with the `--ensemble_size` argument (which is 1 by default). For example:
```
python train.py --data_path data/tox21.csv --dataset_type classification --ensemble_size 5
```

### Model hyperparameters and augmentations

The base message passing architecture can be modified in a range of ways that can be controlled through command line arguments. The full range of options can be seen in `parsing.py`. Suggested modifications are:
* `--hidden_size <int>` Control the hidden size of the neural network layers.
* `--depth <int>` Control the number of message passing steps.
* `--virtual_edges` Adds "virtual" edges connected non-bonded atoms to improve information flow. This works very well on some datasets (ex. QM9) but very poorly on others (ex. delaney).

## Predicting

To load a trained model and make predictions, run `predict.py` and specify:
* `--test_path` Path to the data to predict on.
* `--checkpoint_dir` Directory where the checkpoints were saved (i.e. `--save_dir` during training).
* `--preds_path` Path where a CSV file containing the predictions will be saved.

For example:
```
python predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints --preds_path tox21_preds.csv
```

## TensorBoard

During training, TensorBoard logs are automatically saved to the same directory as the model checkpoints. To view TensorBoard logs, run `tensorboard --logdir=<dir>` where `<dir>` is the path to the checkpoint directory. Then navigate to [http://localhost:6006](http://localhost:6006).

## Deepchem test
We tested our model on 14 deepchem benchmark datasets (http://moleculenet.ai/), ranging from physical chemistry to biophysics
properties. To train our model on those datasets, run:
```
bash run.sh 1
```
where 1 is the random seed for randomly splitting the dataset into training, validation and testing (not applied to datasets with scaffold splitting).

### Results

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
| PCBA | 437,929 | 0.218 ± 0.001* | 	0.136 ± 0.003 | 

Results on regression datasets (score, the lower the better)

Dataset | Size | Ours | GraphConv/MPNN (deepchem) |
| :---: | :---: | :---: | :---: |
delaney	| 1,128 | 0.687 ± 0.037 | 	0.58 ± 0.03 |
Freesolv | 642 |	0.915 ± 0.154	| 1.15 ± 0.12 |
Lipo | 4,200 |	0.565 ± 0.052 |	0.655 ± 0.036 |
qm8 | 21,786 |	0.008 ± 0.000 | 0.0143 ± 0.0011 |
qm9 | 133,885 |	2.47 ± 0.036	| 3.2 ± 1.5 |

†HIV was trained with hidden size 1800 and depth 6 but without the master node.
*MUV and PCBA are using a much older version of the model.
