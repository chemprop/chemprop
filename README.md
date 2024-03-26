![ChemProp Logo](logo/chemprop_logo.svg)
# Chemprop

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/chemprop)](https://badge.fury.io/py/chemprop)
[![PyPI version](https://badge.fury.io/py/chemprop.svg)](https://badge.fury.io/py/chemprop)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/chemprop/badges/version.svg)](https://anaconda.org/conda-forge/chemprop)
[![Build Status](https://github.com/chemprop/chemprop/workflows/tests/badge.svg)](https://github.com/chemprop/chemprop/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/chemprop/badge/?version=latest)](https://chemprop.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/chemprop)](https://pepy.tech/project/chemprop)
[![Downloads](https://static.pepy.tech/badge/chemprop/month)](https://pepy.tech/project/chemprop)
[![Downloads](https://static.pepy.tech/badge/chemprop/week)](https://pepy.tech/project/chemprop)

Chemprop is a repository containing message passing neural networks for molecular property prediction.

**License:** Chemprop is free to use under the [MIT License](LICENSE.txt). The Chemprop logo is free to use under [CC0 1.0](logo/LICENSE.txt).

**References**: Please cite the appropriate papers if Chemprop is helpful to your research.

- Chemprop was initially described in the papers [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) for molecules and [Machine Learning of Reaction Properties via Learned Representations of the Condensed Graph of Reaction](https://doi.org/10.1021/acs.jcim.1c00975) for reactions.
- The interpretation functionality is based on the paper [Multi-Objective Molecule Generation using Interpretable Substructures](https://arxiv.org/abs/2002.03244).
- Chemprop now has its own dedicated manuscript that describes and benchmarks it in more detail: [Chemprop: A Machine Learning Package for Chemical Property Prediction](https://doi.org/10.1021/acs.jcim.3c01250).

**Selected Applications**: Chemprop has been successfully used in the following works.

- [A Deep Learning Approach to Antibiotic Discovery](https://www.cell.com/cell/fulltext/S0092-8674(20)30102-1) - _Cell_ (2020): Chemprop was used to predict antibiotic activity against _E. coli_, leading to the discovery of [Halicin](https://en.wikipedia.org/wiki/Halicin), a novel antibiotic candidate. Model checkpoints are availabile on [Zenodo](https://doi.org/10.5281/zenodo.6527882).
- [Discovery of a structural class of antibiotics with explainable deep learning](https://www.nature.com/articles/s41586-023-06887-8) - _Nature_ (2023): Identified a structural class of antibiotics selective against methicillin-resistant _S. aureus_ (MRSA) and vancomycin-resistant enterococci using ensembles of Chemprop models, and explained results using Chemprop's interpret method.
- [ADMET-AI: A machine learning ADMET platform for evaluation of large-scale chemical libraries](https://www.biorxiv.org/content/10.1101/2023.12.28.573531v1): Chemprop was trained on 41 absorption, distribution, metabolism, excretion, and toxicity (ADMET) datasets from the [Therapeutics Data Commons](https://tdcommons.ai). The Chemprop models in ADMET-AI are available both as a web server at [admet.ai.greenstonebio.com](https://admet.ai.greenstonebio.com) and as a Python package at [github.com/swansonk14/admet_ai](https://github.com/swansonk14/admet_ai). 
- A more extensive list of successful Chemprop applications is given in our [2023 paper](https://doi.org/10.1021/acs.jcim.3c01250)

## Table of Contents

- [Documentation](#documentation)
- [Tutorials and Examples](#tutorials-and-examples)
- [Requirements](#requirements)
- [Installation](#installation)
  * [Option 1: Installing from PyPi](#option-1-installing-from-pypi)
  * [Option 2: Installing from source](#option-2-installing-from-source)
  * [Docker](#docker)
- [Known Issues](#known-issues)
- [Web Interface](#web-interface)
- [Within Python](#within-python)
- [Data](#data)
- [Training](#training)
  * [Train/Validation/Test Splits](#trainvalidationtest-splits)
  * [Loss functions](#loss-functions)
  * [Metrics](#metrics)
  * [Cross validation and ensembling](#cross-validation-and-ensembling)
  * [Aggregation](#aggregation)
  * [Additional Features](#additional-features)
    * [Custom Features](#molecule-level-custom-features)
    * [RDKit 2D Features](#molecule-level-rdkit-2d-features)
    * [Atomic Features](#atom-level-features)
  * [Spectra](#spectra)
  * [Reaction](#reaction)
  * [Reaction in a solvent / Reaction and a molecule](#reaction-in-a-solvent--reaction-and-a-molecule)
  * [Atomic and bond properties prediction](#atomic-and-bond-properties-prediction)
  * [Pretraining](#pretraining)
  * [Missing target values](#missing-target-values)
  * [Weighted training by target and data](#weighted-training-by-target-and-data)
  * [Caching](#caching)
- [Predicting](#predicting)
  * [Uncertainty Estimation](#uncertainty-estimation)
  * [Uncertainty Calibration](#uncertainty-calibration)
  * [Uncertainty Evaluation Metrics](#uncertainty-evaluation-metrics)
- [Hyperparameter Optimization](#hyperparameter-optimization)
  * [Choosing the Search Parameters](#choosing-the-search-parameters)
  * [Checkpoints and Parallel Operation](#checkpoints-and-parallel-operation)
  * [Random or Directed Search](#random-or-directed-search)
  * [Manual Trials](#manual-trials)
- [Encode Fingerprint Latent Representation](#encode-fingerprint-latent-representation)
- [Interpreting Model Prediction](#interpreting)
- [TensorBoard](#tensorboard)
- [Results](#results)

## Documentation

* Documentation of Chemprop is available at https://chemprop.readthedocs.io/en/latest/. Note that this site is several versions behind. An up-to-date version of Read the Docs is forthcoming with the release of Chemprop v2.0.
* This README is currently the best source for documentation on more recently-added features.
* Please also see descriptions of all the possible command line arguments in our [`args.py`](https://github.com/chemprop/chemprop/blob/master/chemprop/args.py) file.

## Tutorials and Examples

* [Benchmark scripts](https://github.com/chemprop/chemprop_benchmark) - scripts from our 2023 paper, providing examples of many features using Chemprop v1.6.1
* [ACS Fall 2023 Workshop](https://github.com/chemprop/chemprop-workshop-acs-fall2023) - presentation, interactive demo, exercises on Google Colab with solution key
* [Google Colab notebook](https://colab.research.google.com/github/chemprop/chemprop/blob/master/colab_demo.ipynb) - several examples, intended to be run in Google Colab rather than as a Jupyter notebook on your local machine
* [nanoHUB tool](https://nanohub.org/resources/chempropdemo/) - a notebook of examples similar to the Colab notebook above, doesn't require any installation
  * [YouTube video](https://www.youtube.com/watch?v=TeOl5E8Wo2M) - lecture accompanying nanoHUB tool
* These [slides](https://docs.google.com/presentation/d/14pbd9LTXzfPSJHyXYkfLxnK8Q80LhVnjImg8a3WqCRM/edit?usp=sharing) provide a Chemprop tutorial and highlight additions as of April 28th, 2020

## Requirements

For small datasets (~1000 molecules), it is possible to train models within a few minutes on a standard laptop with CPUs only. However, for larger datasets and larger Chemprop models, we recommend using a GPU for significantly faster training.

To use `chemprop` with GPUs, you will need:
 * cuda >= 8.0
 * cuDNN

## Installation

Chemprop can either be installed from PyPi via pip or from source (i.e., directly from this git repo). The PyPi version includes a vast majority of Chemprop functionality, but some functionality is only accessible when installed from source.

Both options require conda, so first install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html).

Then proceed to either option below to complete the installation. If installing the environment with conda seems to be taking too long, you can also try running `conda install -c conda-forge mamba` and then replacing `conda` with `mamba` in each of the steps below.

**Note for machines with GPUs:** You may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/). If you're encountering issues with Chemprop not using a GPU on your system after following the instructions below, check which version of PyTorch you have installed in your environment using `conda list | grep torch` or similar. If the PyTorch line includes `cpu`, please uninstall it using `conda remove pytorch` and reinstall a GPU-enabled version using the instructions at the link above.

### Option 1: Installing from PyPi

1. `conda create -n chemprop python=3.8`
2. `conda activate chemprop`
3. `pip install chemprop`

> [!NOTE]  
> Some features that were not made available in the main releases of Chemprop are instead available through 'feature releases' via PyPI: 
> - SSL Pre-train with DDP - available in version `1.6.1.dev0`, install with `pip install chemprop==1.6.1.dev0`.

### Option 2: Installing from source

1. `git clone https://github.com/chemprop/chemprop.git`
2. `cd chemprop`
3. `conda env create -f environment.yml`
4. `conda activate chemprop`
5. `pip install -e .`

### Docker

Chemprop can also be installed with Docker.
Docker makes it possible to isolate the Chemprop code and environment.
You can either pull a pre-built image or build it locally.

Note that regardless of installation method you will need to run the `docker run` command with the `--gpus` command line flag to access GPUs on your machine.

In addition, you will also need to ensure that the CUDA toolkit version in the Docker image is compatible with the CUDA driver on your host machine.
Newer CUDA driver versions are backward-compatible with older CUDA toolkit versions.
To set a specific CUDA toolkit version, add `cudatoolkit=X.Y` to `environment.yml` before building the Docker image.

#### Pull Pre-Built

Run this command to download and run a given release version of Chemprop:

`docker run -it chemprop/chemprop:X.Y.X`

where `X.Y.Z` is the version you want to download, i.e. `1.7.0`.

> [!NOTE]
> Not all versions of Chemprop are available from DockerHub - see the [DockerHub](https://hub.docker.com/r/chemprop/chemprop/tags) page for a complete list of those available.

DockerHub also has a `latest` tag - this is _not_ the latest release of Chemprop, but rather the latest version of `master` which is _not necessarily fit for deployment_.
Use this tag only for development or if you need to access a feature which has not yet been formally released!

#### Local Build

To install and run our code in a Docker container, follow these steps:

1. `git clone https://github.com/chemprop/chemprop.git`
2. `cd chemprop`
3. Install Docker from [https://docs.docker.com/install/](https://docs.docker.com/install/)
4. `docker build -t chemprop .`
5. `docker run -it chemprop:latest`

## Known Issues

As we approach the upcoming release of Chemprop v2.0, we have closed [several issues](https://github.com/chemprop/chemprop/issues?q=label%3Av1-wontfix+) corresponding to bugs that we don't plan to fix before the final release of v1 (v1.7). We will be discontinuing support for v1 once v2 is released, but we still appreciate bug reports and will tag them as [`v1-wontfix`](https://github.com/chemprop/chemprop/issues?q=label%3Av1-wontfix+) so the community can find them easily.

## Web Interface

For those less familiar with the command line, Chemprop also includes a web interface which allows for basic training and predicting. You can start the web interface on your local machine in two ways. Flask is used for development mode while gunicorn is used for production mode.

### Flask

Run `chemprop_web` (or optionally `python web.py` if installed from source) and then navigate to [localhost:5000](http://localhost:5000) in a web browser.

### Gunicorn

Gunicorn is only available for a UNIX environment, meaning it will not work on Windows. It is not installed by default with the rest of Chemprop, so first run:

```
pip install gunicorn
```

Next, navigate to `chemprop/web` and run `gunicorn --bind {host}:{port} 'wsgi:build_app()'`. This will start the site in production mode.
   * To run this server in the background, add the `--daemon` flag.
   * Arguments including `init_db` and `demo` can be passed with this pattern: `'wsgi:build_app(init_db=True, demo=True)'` 
   * Gunicorn documentation can be found [here](http://docs.gunicorn.org/en/stable/index.html).

## Within Python

For information on the use of Chemprop within a python script, refer to the [Within a python script](https://chemprop.readthedocs.io/en/latest/tutorial.html#within-a-python-script)
section of the documentation. A [Google Colab notebook](https://colab.research.google.com/github/chemprop/chemprop/blob/master/colab_demo.ipynb) is also available with several examples. Note that this notebook is intended to be run in Google Colab rather than as a Jupyter notebook on your local machine. A similar notebook of examples is available as a [nanoHUB tool](https://nanohub.org/resources/chempropdemo/).


## Data

In order to train a model, you must provide training data containing molecules (as SMILES strings) and known target values.

Chemprop can either train on a single target ("single tasking") or on multiple targets simultaneously ("multi-tasking").

There are four current supported dataset types. Targets with unknown values can be left as blanks.
* **Regression.** Targets are float values. With bounded loss functions or metrics, the values may also be simple inequalities (e.g., >7.5 or <5.0).
* **Classification.** Targets are binary (i.e. 0s and 1s) indicators of the classification.
* **Multiclass.** Targets are integers (starting with zero) indicating which class the datapoint belongs to, out of a total number of exclusive classes indicated with `--multiclass_num_classes <int>`.
* **Spectra.** Targets are positive float values with each target representing the signal at a specific spectrum position.

The data file must be be a **CSV file with a header row**. For example:
```
smiles,NR-AR,NR-AR-LBD,NR-AhR
CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,1
CCN1C(=O)NC(c2ccccc2)C1=O,0,,0
...
```

By default, it is assumed that the SMILES are in the first column (can be changed using `--number_of_molecules`) and the targets are in the remaining columns. However, the specific columns containing the SMILES and targets can be specified using the `--smiles_columns <column_1> ...` and `--target_columns <column_1> <column_2> ...` flags, respectively.

Datasets from [MoleculeNet](https://moleculenet.org/) and a 450K subset of ChEMBL from [http://www.bioinf.jku.at/research/lsc/index.html](http://www.bioinf.jku.at/research/lsc/index.html) have been preprocessed and are available in `data.tar.gz`. To uncompress them, run `tar xvzf data.tar.gz`.

## Training

To train a model, run:
```
chemprop_train --data_path <path> --dataset_type <type> --save_dir <dir>
```
where `<path>` is the path to a CSV file containing a dataset, `<type>` is one of [classification, regression, multiclass, spectra] depending on the type of the dataset, and `<dir>` is the directory where model checkpoints will be saved.

For example:
```
chemprop_train --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints
```

A full list of available command-line arguments can be found in [chemprop/args.py](https://github.com/chemprop/chemprop/blob/master/chemprop/args.py).

If installed from source, `chemprop_train` can be replaced with `python train.py`.

Notes:
* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with `--metric <metric>`.
* `--save_dir` may be left out if you don't want to save model checkpoints.
* `--quiet` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the `save_dir`.

### Train/Validation/Test Splits

Our code supports several methods of splitting data into train, validation, and test sets.

* **Random.** By default, the data will be split randomly into train, validation, and test sets.
* **Scaffold.** Alternatively, the data can be split by molecular scaffold so that the same scaffold never appears in more than one split. This can be specified by adding `--split_type scaffold_balanced`. Note that the atom-mapped numbers for atom-mapped SMILES will be removed before computing the Bemis-Murcko scaffold.
* **k-Fold Cross-Validation.** A split type specified with `--split_type cv` intended for use when training with cross-validation. The data are split randomly into k groups of equal size, where k is the number of cross-validation folds specified with `--num_folds <k>`. Each group is used once as the test set and once as the validation set in training the k folds of the model. Alternatively, the option `--split_type cv-no-test` can be used to train without a test splits.
* **Random With Repeated SMILES.** Some datasets have multiple entries with the same SMILES. To constrain splitting so the repeated SMILES are in the same split, use the argument `--split_type random_with_repeated_smiles`.
* **Separate val/test.** If you have separate data files you would like to use as the validation or test set, you can specify them with `--separate_val_path <val_path>` and/or `--separate_test_path <test_path>`. If both are provided, then the data specified by `--data_path` is used entirely as the training data. If only one separate path is provided, the `--data_path` data is split between train data and either val or test data, whichever is not provided separately.

When data contains multiple molecules per datapoint, scaffold and repeated SMILES splitting will only constrain splitting based on one of the molecules. The key molecule can be chosen with the argument `--split_key_molecule <int>`, with the default setting using an index of 0 indicating the first molecule.

By default, both random and scaffold split the data into 80% train, 10% validation, and 10% test. This can be changed with `--split_sizes <train_frac> <val_frac> <test_frac>`. The default setting is `--split_sizes 0.8 0.1 0.1`. If a separate validation set or test set is provided, the split defaults to 80%-20%. Splitting involves a random component and can be seeded with `--seed <seed>`. The default setting is `--seed 0`. The split size argument is not used with split types `cv` or `cv-no-test`.

### Loss functions

The loss functions available for training are dependent on the selected dataset type. Loss functions other than the defaults can be selected from the supported options with the argument `--loss_function <function>`.
* **Regression.** mse (default), bounded_mse, mve (mean-variance estimation, a.k.a. heteroscedastic loss), evidential, quantile_interval (Pinball loss, specify margins with `--quantile_loss_alpha <float>`).
* **Classification.** binary_cross_entropy (default), mcc (a soft version of Matthews Correlation Coefficient), dirichlet (a.k.a. evidential classification)
* **Multiclass.** cross_entropy (default), mcc (a soft version of Matthews Correlation Coefficient)
* **Spectra.** sid (default, spectral information divergence), wasserstein (First-order Wasserstein distance a.k.a. earthmover's distance.)


Dropout regularization can be applied regardless of loss function using the argument `--dropout <float>` and providing a dropout fraction between 0 and 1.

The regression loss functions `mve` and `evidential` function by minimizing the negative log likelihood of a predicted uncertainty distribution. If used during training, the uncertainty predictions from these loss functions can be used for uncertainty prediction during prediction tasks. A regularization specific to evidential learning can be applied using the argument `--evidential_regularization <float>`. The regression loss function `quantile_interval` trains the model with two different output heads which correspond to the `quantile_loss_alpha/2` and `1 - quantile_loss_alpha/2` quantile predictions. Since it is a symmetrical interval, return the center of the interval as the predicted value. The evaluation metric for `quantile_interval` is automatically set to the `quantile` metric.

### Metrics

Metrics are used to evaluate the success of the model against the test set as the final model score and to determine the optimal epoch to save the model at based on the validation set. The primary metric used for both purposes is selected with the argument `--metric <metric>` and additional metrics for test set score only can be added with `--extra_metrics <metric1> <metric2> ...`. Supported metrics are dependent on the dataset type. Unlike loss functions, metrics do not have to be differentiable.
* **Regression.** rmse (default), mae, mse, r2, bounded_rmse, bounded_mae, bounded_mse (default if bounded_mse is loss function), quantile (average of pinball loss for both output heads).
* **Classification.** auc (default), prc-auc, accuracy, binary_cross_entropy, f1, mcc, recall, precision and balanced accuracy.
* **Multiclass.** cross_entropy (default), accuracy, f1, mcc.
* **Spectra.** sid (default), wasserstein.

When a multitask model is used, the metric score used for evaluation at each epoch or for choosing the best set of hyperparameters during hyperparameter search is obtained by taking the mean of the metric scores for each task. Some metrics scale with the magnitude of the targets (most regression metrics), so geometric mean instead of arithmetic mean is used in those cases in order to avoid having the mean score dominated by changes in the larger magnitude task.
### Cross validation and ensembling

Cross-validation can be run by specifying `--num_folds <k>`. The default is `--num_folds 1`. Each trained model will have different train/val/test splits, determined according to the specified split type argument and split sizes argument but using a different random seed to perform the splitting. The reported test score will be the average of the metrics from each fold. To use a strict k-fold cross-validation where each datapoint will appear in fold test sets exactly once, the argument `--split_type cv` must be used.

To train an ensemble, specify the number of models in the ensemble with `--ensemble_size <n>`. The default is `--ensemble_size 1`. Each trained model within the ensemble will share data splits. The reported test score for one ensemble is the metric applied to the averaged prediction across the models. Ensembling and cross-validation can be used at the same time.

### Aggregation

By default, the atom-level representations from the message passing network are averaged over all atoms of a molecule to yield a molecule-level representation. Alternatively, the atomic vectors can be summed up (by specifying `--aggregation sum`) or summed up and divided by a constant number N (by specifying `--aggregation norm --aggregation_norm <N>`). A reasonable value for N is usually the average number of atoms per molecule in the dataset of interest. The default is `--aggregation_norm 100`.

### Additional Features

While the model works very well on its own, especially after hyperparameter optimization, we have seen that additional features can further improve performance on certain datasets. The additional features can be added at the atom-, bond, or molecule-level. Molecule-level features can be either automatically generated by RDKit or custom features provided by the user.

#### Molecule-Level Custom Features

If you install from source, you can modify the code to load custom features as follows:

1. **Generate features:** If you want to generate features in code, you can write a custom features generator function in `chemprop/features/features_generators.py`. Scroll down to the bottom of that file to see a features generator code template.
2. **Load features:** If you have features saved as a numpy `.npy` file or as a `.csv` file, you can load the features by using `--features_path /path/to/features`. Note that the features must be in the same order as the SMILES strings in your data file. Also note that `.csv` files must have a header row and the features should be comma-separated with one line per molecule. By default, provided features will be normalized unless the flag `--no_features_scaling` is used.

#### Molecule-Level RDKit 2D Features

As a starting point, we recommend using pre-normalized RDKit features by using the `--features_generator rdkit_2d_normalized --no_features_scaling` flags. In general, we recommend NOT using the `--no_features_scaling` flag (i.e. allow the code to automatically perform feature scaling), but in the case of `rdkit_2d_normalized`, those features have been pre-normalized and don't require further scaling. The utilization of the `rdkit_2d_normalized` should be avoided in cases where molecule-level custom features have been loaded and necessitate additional scaling.

The full list of available features for `--features_generator` is as follows. 

`morgan` is binary Morgan fingerprints, radius 2 and 2048 bits.
`morgan_count` is count-based Morgan, radius 2 and 2048 bits.
`rdkit_2d` is an unnormalized version of 200 assorted rdkit descriptors. Full list can be found at the bottom of our paper: https://arxiv.org/pdf/1904.01561.pdf
`rdkit_2d_normalized` is the CDF-normalized version of the 200 rdkit descriptors.

#### Atom-Level Features

Similar to the additional molecular features described above, you can also provide additional atomic features via `--atom_descriptors_path /path/to/features` with valid file formats:
* `.npz` file, where descriptors are saved as 2D array for each molecule in the exact same order as the SMILES strings in your data file.
* `.pkl` / `.pckl` / `.pickle` containing a pandas dataframe with smiles as index and a numpy array of descriptors as columns.
* `.sdf` containing all mol blocks with descriptors as entries.

The order of the descriptors for each atom per molecule must match the ordering of atoms in the RDKit molecule object. Further information on supplying atomic descriptors can be found [here](https://github.com/chemprop/chemprop/releases/tag/v1.1.0). 

Users must select in which way atom descriptors are used. The command line option `--atom_descriptors descriptor` concatenates the new features to the embedded atomic features after the D-MPNN with an additional linear layer. The option `--atom_descriptors feature` concatenates the features to each atomic feature vector before the D-MPNN, so that they are used during message-passing. Alternatively, the user can overwrite the default atom features with the custom features using the option `--overwrite_default_atom_features`. 

Similar to the molecule-level features, the atom-level descriptors and features are scaled by default. This can be disabled with the option `--no_atom_descriptor_scaling`

#### Bond-Level Features

Bond-level features can be provided in the same format as the atom-level features, using the option `--bond_descriptors_path /path/to/features`. The order of the features for each molecule must match the bond ordering in the RDKit molecule object.

Users must select in which way bond descriptors are used. The command line option `--bond_descriptors feature` concatenates the bond-level features with the bond feature vectors before the D-MPNN, such that they are used during message-passing. For atomic/bond properties prediction, the command line option `--bond_descriptors descriptor` concatenates the new features to the embedded bond features after the D-MPNN with an additional linear layer. Alternatively, the user can overwrite the default bond features with the custom features using the option `--overwrite_default_bond_features`.

Similar to molecule-level and atom-level features, the bond-level descriptors and features are scaled by default. This can be disabled with the option `--no_bond_descriptor_scaling`.

### Spectra

One of the data types that can be trained with Chemprop is "spectra". Spectra training is different than other datatypes because it considers the predictions of all targets together. Targets for spectra should be provided as the values for the spectrum at a specific position in the spectrum. The loss function for spectra is SID, spectral information divergence. Alternatively, Wasserstein distance (earthmover's distance) can be used for both loss function and metric with input arguments `--metric wasserstein --loss_function wasserstein`.

Spectra predictions are configured to return only positive values and normalize them to sum each spectrum to 1. Activation to enforce positivity is an exponential function by default but can also be set as a Softplus function, according to the argument `--spectra_activation <exp or softplus>`. Value positivity is enforced on input targets as well using a floor value that replaces negative or smaller target values with the floor value (default 1e-8), customizable with the argument `--spectra_target_floor <float>`.

In absorption spectra, sometimes the phase of collection will create regions in the spectrum where data collection or prediction would be unreliable. To exclude these regions, include paths to phase features for your data (`--phase_features_path <path>`) and a mask indicating the spectrum regions that are supported (`--spectra_phase_mask_path <path>`). The format for the mask file is a `.csv` file with columns for the spectrum positions and rows for the phases, with column and row labels in the same order as they appear in the targets and features files.

### Reaction

As an alternative to molecule SMILES, Chemprop can also process atom-mapped reaction SMILES (see [Daylight manual](https://www.daylight.com/meetings/summerschool01/course/basics/smirks.html) for details on reaction SMILES), which consist of three parts denoting reactants, agents and products, separated by ">". Use the option `--reaction` to enable the input of reactions, which transforms the reactants and products of each reaction to the corresponding condensed graph of reaction and changes the initial atom and bond features to hold information from both the reactant and product (option `--reaction_mode reac_prod`), or from the reactant and the difference upon reaction (option `--reaction_mode reac_diff`, default) or from the product and the difference upon reaction (option `--reaction_mode prod_diff`). In reaction mode, Chemprop thus concatenates information to each atomic and bond feature vector, for example, with option `--reaction_mode reac_prod`, each atomic feature vector holds information on the state of the atom in the reactant (similar to default Chemprop), and concatenates information on the state of the atom in the product, so that the size of the D-MPNN increases slightly. Agents are discarded. Functions incompatible with a reaction as input (scaffold splitting and feature generation) are carried out on the reactants only. If the atom-mapped reaction SMILES contain mapped hydrogens, enable explicit hydrogens via `--explicit_h`. Example of an atom-mapped reaction SMILES denoting the reaction of methanol to formaldehyde without hydrogens: `[CH3:1][OH:2]>>[CH2:1]=[O:2]` and with hydrogens: `[C:1]([H:3])([H:4])([H:5])[O:2][H:6]>>[C:1]([H:3])([H:4])=[O:2].[H:5][H:6]`. The reactions do not need to be balanced and can thus contain unmapped parts, for example leaving groups, if necessary. With reaction modes `reac_prod`, `reac_diff` and `prod_diff`, the atom and bond features of unbalanced aroma are set to zero on the side of the reaction they are not specified. Alternatively, features can be set to the same values on the reactant and product side via the modes `reac_prod_balance`, `reac_diff_balance` and `prod_diff_balance`, which corresponds to a rough balancing of the reaction.
For further details and benchmarking, as well as a citable reference, please refer to the [article](https://doi.org/10.1021/acs.jcim.1c00975).

### Reaction in a solvent / Reaction and a molecule

Chemprop can process a reaction in a solvent or a reaction and a molecule with the `--reaction_solvent` option. While this
option is originally built to model a reaction in a solvent, this option works for any reaction and a molecule where 
the molecule can represent anything, i.e. a solvent, a reagent, etc.
This requires the input csv file to have two separate columns of SMILES: one column for atom-mapped reaction SMILES 
and the other column for solvent/molecule SMILES. The reaction and solvent/molecule SMILES columns can be ordered in 
any way (i.e. the first column can be either reaction SMILES or solvent SMILES and the second column can then be 
solvent SMILES or reaction SMILES). However, the same column ordering as used in the training must be used for the prediction
(i.e. if the input csv file used for model training had reaction SMILES as the first column and solvent SMILES as the 
second columns, the csv file used for prediction should also have the first column as reaction SMILES and second column 
as the solvent SMILES). For the information on atom-mapped reaction SMILES, please refer to [Reaction](#reaction).

When using the `--reaction_solvent` option, `--number_of_molecules` must be set to 2. All options listed in the [Reaction](#reaction) 
section such as different `--reaction_mode` and `--explicit_h` can be used for `--reaction_solvent`. Note that 
`--explicit_h` option is only applicable to reaction SMILES. The `--adding_h` option can be used instead for 
solvent/molecule if one wishes to add hydrogens to solvent/molecule SMILES. Chemprop allows differently sized MPNNs to be used for each 
reaction and solvent/molecule encoding. Below are the input arguments for specifying the size and option of the two MPNNs:
* Reaction:
  * `--bias` Whether to add bias to linear layers.
  * `--hidden_size` Dimensionality of hidden layers.
  * `--depth` Number of message passing steps.
  * `--explicit_h` Whether H are explicitly specified in input and should be kept this way. Only applicable to reaction SMILES.
* Solvent / Molecule:
  * `--bias_solvent` Whether to add bias to linear layers for solvent/molecule MPN.
  * `--hidden_size_solvent` Dimensionality of hidden layers in solvent/molecule MPN.
  * `--depth_solvent` Number of message passing steps for solvent/molecule.
  * `--adding_h` Whether RDKit molecules will be constructed with adding the Hs to them. Applicable to any SMILES that is not reaction.

### Atomic and bond properties prediction

Chemprop can perform multitask constrained message passing neural networks for atomic/bond properties prediction as described in this [paper](https://chemrxiv.org/articles/preprint/Regio-Selectivity_Prediction_with_a_Machine-Learned_Reaction_Representation_and_On-the-Fly_Quantum_Mechanical_Descriptors/12907316). This model can train on any number of atomic/bond properties simultaneously. In the original work, a total loss was calculated as a weighted sum of every single loss, where the weights were required to be specified for the regression task. In this repository, these weights have been automatically taken into account by doing standardization of all the training targets. In order to train a model, training data containing molecules (as SMILES strings) and known atomic/bond target values are required, and the `--is_atom_bond_targets` flag is used. The input is a csv file. For example:
```
                              smiles                                  hirshfeld_charges  ...                                 bond_length_matrix                                  bond_index_matrix
0     CNC(=S)N/N=C/c1c(O)ccc2ccccc12  [-0.026644, -0.075508, 0.096217, -0.287798, -0...  ...  [[0.0, 1.4372890960937539, 2.4525543850909814,...  [[0.0, 0.9595, 0.0158, 0.0162, 0.0103, 0.0008,...
1      O=C(NCCn1cccc1)c1cccc2ccccc12  [-0.292411, 0.170263, -0.085754, 0.002736, 0.0...  ...  [[0.0, 1.2158509801073485, 2.2520730233154076,...  [[0.0, 1.6334, 0.1799, 0.0086, 0.0068, 0.0002,...
2  C=C(C)[C@H]1C[C@@H]2OO[C@H]1C=C2C  [-0.101749, 0.012339, -0.07947, -0.020027, -0....  ...  [[0.0, 1.3223632546838255, 2.468055985361353, ...  [[0.0, 1.9083, 0.0179, 0.016, 0.0236, 0.001, 0...
3                     OCCCc1cc[nH]n1  [-0.268379, 0.027614, -0.050745, -0.045047, 0....  ...  [[0.0, 1.4018301850170725, 2.4667588956616737,...  [[0.0, 0.9446, 0.0311, 0.002, 0.005, 0.0007, 0...
4      CC(=N)NCc1cccc(CNCc2ccncc2)c1  [-0.083162, 0.114954, -0.274544, -0.100369, 0....  ...  [[0.0, 1.5137126697008916, 2.4882198180715465,...  [[0.0, 1.0036, 0.0437, 0.0108, 0.0134, 0.0004,......
```
where atomic properties (e.g. hirshfeld_charges) must be a 1D list with the order same as that of atoms in the SMILES string; and bond properties (e.g. bond_length_matrix) can either be a 2D list of shape (number_of_atoms × number_of_atoms) or a 1D list with the order same as that of bonds in the SMILES string. The `--keeping_atom_map` option can be used if atom-mapped SMILES is provided. The `--adding_h` option can be used if hydrogens are included in the atom targets and bonds to hydrogens are included in the bond targets.
This model allows multitask constraints applied to different atomic/bond properties by specifying the argument `--constraints_path` with a given `.csv` file. Note that the constraints must be in the same order as the SMILES strings in your data file. Also note that `.csv` file must have a header row and the constraints should be comma-separated with one line per molecule. The optional argument `--no_shared_atom_bond_ffn` will make it so that the ffn weights used by each task are independent, otherwise the default is that atom tasks share ffn weights and bond tasks share ffn weights so that the ffn weights have the benefits of multitask training. The optional argument `--no_adding_bond_types` will let the bond types of each bond determined by RDKit molecules not be added to the output of bond targets. The optional argument `--weights_ffn_num_layers` can change the number of layers in FFN for determining weights used to correct the constrained targets.

Please note that the current framework is only available for models trained on multiple atomic and bond properties simultaneously. Training on both atomic/bond and molecular targets is not supported.

### Pretraining

Pretraining can be carried out using previously trained checkpoint files to set some or all of the initial values of a model for training. Additionally, some model parameters from the previous model can be frozen in place, so that they will not be updated during training.

Parameters from existing models can be used for parameter-initialization of a new model by providing a checkpoint of the existing model using either
 * `--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. `--save_dir` during training of the old model). This will walk the directory, and load all `.pt` files it finds.
 * `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
 * `--checkpoint_paths <list of paths>` A list of paths to multiple model checkpoint (`.pt`) files.
when training the new model. The model architecture of the new model should resemble the architecture of the old model - otherwise some or all parameters might not be loaded correctly. If any of these options are specified during training, any argument provided with `--ensemble_size` will be overwritten and the ensemble size will be specified as the number of checkpoint files that were provided, with each submodel in the ensemble using a separate checkpoint file for initialization. When using these options, new model parameters are initialized using the old checkpoint files but all parameters remain trainable (no frozen layers from these arguments).

Certain portions of the model can be loaded from a previous model and frozen so that they will not be trainable, using the various frozen layer parameters. A path to a checkpoint file for frozen parameters is provided with the argument `--checkpoint_frzn <path>`. If this path is provided, the parameters in the MPNN portion of the model will be specified from the path and frozen. Layers in the FFNN portion of the model can also be applied and frozen in addition to freezing the MPNN using `--frzn_ffn_layers <number-of-layers>`. Model architecture of the new model should match the old model in any layers that are being frozen, but non-frozen layers can be different without affecting the frozen layers (e.g., MPNN alone is frozen and new model has a larger number of FFNN layers). Parameters provided with `--checkpoint_frzn` will overwrite initialization parameters from `--checkpoint_path` (or similar) that are frozen in the new model. At present, only one checkpoint can be provided for the `--checkpoint_frzn` and those parameters will be used for any number of submodels if `--ensemble_size` is specified. If multiple molecules (with multiple MPNNs) are being trained in the new model, the default behavior is for both of the new MPNNs to be frozen and drawn from the checkpoint. Only the first MPNN will be frozen and subsequent MPNNs still allowed to train if `--freeze_first_only` is specified.

### Missing Target Values

When training multitask models (models which predict more than one target simultaneously), sometimes not all target values are known for all molecules in the dataset. Chemprop automatically handles missing entries in the dataset by masking out the respective values in the loss function, so that partial data can be utilized, too. The loss function is rescaled according to all non-missing values, and missing values furthermore do not contribute to validation or test errors. Training on partial data is therefore possible and encouraged (versus taking out datapoints with missing target entries). No keyword is needed for this behavior, it is the default.

In contrast, when using `sklearn_train.py` (a utility script provided within Chemprop that trains standard models such as random forests on Morgan fingerprints via the python package scikit-learn), multi-task models cannot be trained on datasets with partially missing targets. However, one can instead train individual models for each task (via the argument `--single_task`), where missing values are automatically removed from the dataset. Thus, the training still makes use of all non-missing values, but by training individual models for each task, instead of one model with multiple output values. This restriction only applies to sklearn models (via  :code:`sklearn_train` or :code:`python sklearn_train.py`), but NOT to default Chemprop models via `chemprop_train` or `python train.py`. Alternatively, missing target values can be imputed by specifying `--impute_mode <single_task/linear/median/mean/frequent>`. The option `single_task` trains single task sklearn models on each task to predict missing values and is computationally expensive. The option `linear` trains a stochastic gradient linear model on each target to compute missing targets. Both `single_task` and `linear` are applicable to regression and classification task. For regression tasks, the options `median` and `mean` furthermore compute the median and mean of the training data. For classification tasks, `frequent` computes the most frequent value for each task. For all options, models are fitted to non-missing training targets and predict missing training targets. The test set is not affected by imputing.

### Weighted Training by Target and Data

By default, each task in multitask training and each provided datapoint are weighted equally for training. Weights can be specified in either case to allow some tasks in training or some specified data points to be weighted more heavily than others in the training of the model.

Using the `--target_weights` argument followed by a list of numbers equal in length to the number of tasks in multitask training, different tasks can be given more weight in parameter updates during training. For instance, in a multitask training with two tasks, the argument `--target_weights 1 2` would give the second task twice as much weight in model parameter updates. Provided weights must be non-negative. Values are normalized to make the average weight equal 1. Target weights are not used with the validation set for the determination of early stopping or in evaluation of the test set.

Using the `--data_weights_path` argument followed by a path to a data file containing weights will allow each individual datapoint in the training data to be given different weight in parameter updates. Formatting of this file is similar to provided features CSV files: they should contain only a single column with one header row and a numerical value in each row that corresponds to the order of datapoints provided with `--data_path`. Data weights should not be provided for validation or test sets if they are provided through the arguments `--separate_test_path` or `--separate_val_path`. Provided weights must be non-negative. Values are normalized to make the average weight equal 1. Data weights are not used with the validation set for the determination of early stopping or in evaluation of the test set.

### Caching

By default, the molecule objects created from each SMILES string are cached for all dataset sizes, and the graph objects created from each molecule object are cached for datasets up to 10000 molecules. If memory permits, you may use the keyword `--cache_cutoff inf` to set this cutoff from 10000 to infinity to always keep the generated graphs in cache (or to another integer value for custom behavior). This may speed up training (depending on the dataset size, molecule size, number of epochs and GPU support), since the graphs do not need to be recreated each epoch, but increases memory usage considerably. Below the cutoff, graphs are created sequentially in the first epoch. Above the cutoff, graphs are created in parallel (on `--num_workers <int>` workers) for each epoch. If training on a GPU, training without caching and creating graphs on the fly in parallel is often preferable. On CPU, training with caching if often preferable for medium-sized datasets and a very low number of CPUs. If a very large dataset causes memory issues, you might turn off caching even of the molecule objects via the commands `--no_cache_mol` to reduce memory usage further.

## Predicting

To load a trained model and make predictions, run `predict.py` and specify:
* `--test_path <path>` Path to the data to predict on.
* A checkpoint by using either:
  * `--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. `--save_dir` during training). This will walk the directory, load all `.pt` files it finds, and treat the models as an ensemble.
  * `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
* `--preds_path` Path where a CSV file containing the predictions will be saved.

For example:
```
chemprop_predict --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints --preds_path tox21_preds.csv
```
or
```
chemprop_predict --test_path data/tox21.csv --checkpoint_path tox21_checkpoints/fold_0/model_0/model.pt --preds_path tox21_preds.csv
```

Predictions made on an ensemble of models will return the average of the individual model predictions. To return the individual model predictions as well, include the `--individual_ensemble_predictions` argument.

If installed from source, `chemprop_predict` can be replaced with `python predict.py`.

### Uncertainty Estimation

The uncertainty of predictions made in Chemprop can be estimated by several different methods. Uncertainty estimation is carried out alongside model value prediction and reported in the predictions csv file when the argument `--uncertainty_method <method>` is provided. If no uncertainty method is provided, then only the model value predictions will be carried out. The available methods are:

* `ensemble` For a prediction using an ensemble of models. Returns the variance of predictions made by each of the ensemble submodels. Ensemble variance can be used with any dataset type, but the results are only usable for calibration or evaluation with regression datasets.
* `dropout` Intended for use with a single model and not an ensemble. This method uses Monte Carlo dropout to generate a virtual ensemble of models and reports the ensemble variance of the predictions. The number of models generated and the probability of dropout can be changed using `--uncertainty_dropout_p <float>` and `--dropout_sampling_size <int>`, respectively. Note that this dropout is distinct from dropout regularization used during training, which is not active during predictions.
* `mve` When mve has been used for the training loss function on regression datasets, this method uses the separate variance prediction of the model. The variance result from ensembling models together includes the variance contribution of the different models having different mean predictions.
* `evidential_total`, `evidential_epistemic`, `evidential_aleatoric` When evidential was used as the training loss function for regression datasets, these methods use the variance prediction of the model. The evidential output includes different functions intended to divide the variance into epistemic and aleatoric uncertainty. The variance result from ensembling models together includes the variance contribution of the different models having different mean predictions.
* `spectra_roundrobin` For an ensemble of spectra predictions. Calculates the pairwise SID between the predictions made by each of the ensemble submodels. Returns the average SID.
* `classification` The predictions of classification and multiclass dataset types are inherently probabilistic already. Used by default for classification and multiclass as needed.

### Uncertainty Calibration

Uncertainty predictions may be calibrated to improve their performance on new predictions. Calibration methods are selected using `--calibration_method <method>`, options provided below. An additional dataset to use in calibration is provided through `--calibration_path <path>`, along with necessary features like `--calibration_features_path <path>`. As with the data used in training, calibration data for multitask models are allowed to have gaps and missing targets in the data.

**Regression** 

Calibrated regression outputs can be in the form of a standard deviation or an interval, as specified with the argument `--regression_calibrator_metric <"stdev" or "interval">`. The interval can be set using `--calibration_interval_percentile <float>` in the range (1,100). The options mentioned above do not apply to the calibration methods `conformal_regression` and `conformal_quantile_regression`.
* `zscaling` Assumes that errors are normally distributed according to the estimated variance for each prediction. Applies a constant multiple to all stdev or interval outputs in order to minimize the negative log likelihood for the normal distributions. (https://arxiv.org/abs/1905.11659)
* `tscaling` Similar to zscaling. Assumes that the errors are normally distributed, but accounts for the ensemble size and uncertainty in the sample variance by using a sample-size reduced t-distribution in the negative log likelihood. Works best when errors are mostly due to variability between model instances and not dataset noise or model bias.
* `zelikman_interval` Assumes that the error distribution is the same for each prediction but scaled by the uncalibrated standard deviation for each. Multiplies the uncalibrated standard deviation by a factor necessary to cover the specified interval of the calibration set. Does not assume a Gaussian distribution. Intended for use with intervals but can return a stdev as well. (https://arxiv.org/abs/2005.12496)
* `mve_weighting` For use with ensembles of models trained with mve or evidential loss function. Uses a weighted average of the predicted variances to achieve a minimum negative log likelihood of predictions. (https://doi.org/10.1186/s13321-021-00551-x)
* `conformal_regression` Generates a symmetric interval of fixed size for each prediction such that the actual value has probability $1-\alpha$ of falling in the interval. The desired error rate is controlled using the parameter `--conformal_alpha <float>` which is set by default to 0.1. (https://arxiv.org/abs/2107.07511)
* `conformal_quantile_regression` Similar to `conformal_regression` but generates an interval of variable size for each prediction based on quantile predictions of the data. The model should be trained with parameters `--loss_function quantile_interval` and `--quantile_loss_alpha <float>` where $\alpha$ is the desired error rate of the quantile interval. The trained model will output the center of the $\alpha/2$ and $1-\alpha/2$ quantiles according to pinball loss as the predicted value and return the half range of the interval as the uncertainty quantification. The parameter `--conformal_alpha <float>` should be included to specify the desired error rate of the conformal method during inference. (https://arxiv.org/abs/2107.07511)

**Classification**
* `platt` Uses a linear scaling before the sigmoid function in prediction to minimize the negative log likelihood of the predictions. If the model checkpoint was generated after Chemprop v1.5.0, then a Bayesian correction is applied to account for the class balance in the training set during prediction. Implemented for classification but not multiclass datasets. (https://arxiv.org/abs/1706.04599)
* `isotonic` Fits an isotonic regression model to the predictions. Prediction outputs are transformed using a stepped histogram-style to match the empirical probability observed in the calibration data. Number and size of the histogram bins are procedurally decided. Histogram bins are wider in the regions of the model output that are less reliable in ordering confidence. Implemented for both classification and multiclass datasets. (https://arxiv.org/abs/1706.04599)
* `conformal` Generates a pair of sets of labels $C_{in} \subset C_{out}$ such that the true set of labels $S$ satisfies the property $C_{in} \subset S \subset C_{out}$ with probability at least $1-\alpha$. The desired error rate $\alpha$ can be controlled with the parameter `--conformal_alpha <float>` which is set by default to 0.1. (https://arxiv.org/abs/2004.10181)

**Multiclass**
* `conformal` Generates a set of possible classes for each prediction such that the true class has probability $1-\alpha$ of falling in the set. The desired error rate $\alpha$ can be controlled with the parameter `--conformal_alpha <float>` which is set by default to 0.1. Set generated using the basic conformal method. (https://arxiv.org/abs/2107.07511)
* `conformal_adaptive` Generates a set of possible classes for each prediction such that the true class has probability 1-alpha of falling in the set. The desired error rate $\alpha$ can be controlled with the parameter `--conformal_alpha <float>` which is set by default to 0.1. Set generated using the adaptive conformal method. (https://arxiv.org/abs/2107.07511)

### Uncertainty Evaluation Metrics

The performance of uncertainty predictions (calibrated or uncalibrated) as evaluated on the test set using different evaluation metrics as specified with `--evaluation_methods <[methods]>`. Evaluation scores will be saved at the path provided with `--evaluation_scores_path <path.csv>`. If no path is provided to save the scores, then the results will only appear in the output trace. Multiple evaluation methods can be provided and they will be calculated separately for each model task. Evaluation is only available when the target values are provided with the data in `--test_path <path.csv>`. As with the data used in training, evaluation data for multitask models are allowed to have gaps and missing targets in the data.

* Any valid classification or multiclass metric. Because classification and multiclass outputs are inherently probabilistic, any metric used to assess them during training is appropriate to evaluate the confidences produced after calibration.
* `nll` Returns the average negative log likelihood of the real target as indicated by the uncertainty predictions. Enabled for regression, classification, and multiclass dataset types.
* `spearman` A regression evaluation metric. Returns the Spearman rank correlation between the predicted uncertainty and the actual error in predictions. Only considers ordering, does not assume a particular probability distribution.
* `ence` Expected normalized calibration error. A regression evaluation metric. Bins model prediction according to uncertainty prediction and compares the RMSE in each bin versus the expected error based on the predicted uncertainty variance then scaled by variance. (discussed in https://doi.org/10.1021/acs.jcim.9b00975)
* `miscalibration_area` A regression evaluation metric. Calculates the model's performance of expected probability versus realized probability at different points along the probability distribution. Values range (0, 0.5) with perfect calibration at 0. (discussed in https://doi.org/10.1021/acs.jcim.9b00975)
* `conformal_coverage` Measures the empirical coverage of the conformal methods, that is the proportion of datapoints that fall within the output set or interval. Must be used with a conformal calibration method which outputs a set or interval. The metric can be used with multiclass, multilabel, or regression conformal methods.

Different evaluation metrics consider different aspects of uncertainty. It is often appropriate to consider multiple metrics. For intance, miscalibration error is important for evaluating uncertainty magnitude but does not indicate that the uncertainty function discriminates well between different outputs. Similarly, spearman tests ordering but not prediction magnitude.

Evaluations can be used to compare different uncertainty methods and different calibration methods for a given dataset. Using evaluations to compare between datasets may not be a fair comparison and should be done cautiously.

## Hyperparameter Optimization

Although the default message passing architecture works well on a variety of datasets, optimizing the hyperparameters for a particular dataset often leads to improvement in performance. We have automated hyperparameter optimization via Bayesian optimization (using the [hyperopt](https://github.com/hyperopt/hyperopt) package). The default hyperparameter optimization will search for the best configuration of hidden size, depth, dropout, and number of feed-forward layers for our model. Optimization can be run as follows:
```
chemprop_hyperopt --data_path <data_path> --dataset_type <type> --num_iters <int> --config_save_path <config_path>
```
where `<int>` is the number of hyperparameter trial configurations to try and `<config_path>` is the path to a `.json` file where the optimal hyperparameters will be saved. If installed from source, `chemprop_hyperopt` can be replaced with `python hyperparameter_optimization.py`. Additional training arguments can also be supplied during submission, and they will be applied to all included training iterations (`--epochs`, `--aggregation`, `--num_folds`, `--gpu`, `--ensemble_size`, `--seed`, etc.). The argument `--log_dir <dir_path>` can optionally be provided to set a location for the hyperparameter optimization log.

Once hyperparameter optimization is complete, the optimal hyperparameters can be applied during training by specifying the config path as follows:
```
chemprop_train --data_path <data_path> --dataset_type <type> --config_path <config_path>
```

Note that the hyperparameter optimization script sees all the data given to it. The intended use is to run the hyperparameter optimization script on a dataset with the eventual test set held out. If you need to optimize hyperparameters separately for several different cross validation splits, you should e.g. set up a bash script to run hyperparameter_optimization.py separately on each split's training and validation data with test held out.

### Choosing the Search Parameters

The parameter space being searched can be changed to include different sets of model hyperparameters. These can be selected using the argument `--search_parameter_keywords <list-of-keywords>`. The available keywords are listed below. Some keywords refer to bundles of parameters or other special behavior. Note that the search ranges for each parameter is hardcoded and can be viewed or changed in `chemprop/hyperopt_utils.py`.

Special keywords
* basic - the default set of hyperparameters for search: depth, ffn_num_layers, dropout, and linked_hidden_size.
* linked_hidden_size - search for hidden_size and ffn_hidden_size, but constrained for them to have the same value. This allows search through both but with one fewer degree of freedom.
* learning_rate - search for max_lr, init_lr, final_lr, and warmup_epochs.
* all - include search for all inidividual keyword options
Individual supported parameters
* activation, aggregation, aggregation_norm, batch_size, depth, dropout, ffn_hidden_size, ffn_num_layers, final_lr, hidden_size, init_lr, max_lr, warmup_epochs

Choosing to include additional search parameters should be undertaken carefully. The number of possible parameter combinations increases combinatorially with the addition of more hyperparameters, so the search for an optimal configuration will become more difficult accordingly. The recommendation from Hyperopt is to use at least 10 trials per hyperparameter for an appropriate search as a rule of thumb, but even more will be necessary at higher levels of search complexity or to obtain better convergence to the optimal hyperparameters. Steps to reduce the complexity of a search space should be considered, such as excluding low-sensitivity parameters or those for which a judgement can be made ahead of time. Splitting the search into two steps can also reduce overall complexity. The `all` search option should only be used in situations where the dataset is small and a very large number of trials can be used.

For best results, the `--epochs` specified during hyperparameter search should be the same as in the intended final application of the model. Learning rate parameters are especially sensitive to the number of epochs used. Note that the number of epochs is not a hyperparameter search option.

The search space for init_lr and final_lr values are defined as fractions of the max_lr value. The search space for warmup_epochs is set by fraction of the `--epochs` training argument. The search for aggregation_norm values is only relevant when the aggregation function is set to norm and can otherwise be neglected. If a separate training argument is provided that is included in the search parameters, the search will overwrite the specified value (e.g., `--depth 5 --search_parameter_keywords depth`).

### Checkpoints and Parallel Operation

Results of completed trial configurations will be stored there and may serve as checkpoints for other instances of hyperparameter optimization if the directory for hyperopt checkpoint files has been specified, `--hyperopt_checkpoint_dir <path>`. If `--hyperopt_checkpoint_dir` is not specified, then checkpoints will default to being stored with the hyperparame. Interrupted hyperparameter optimizations can be restarted by specifying the same directory. Previously completed hyperparameter optimizations can be used as the starting point for new optimizations with a larger selected number of iterations. Note that the `--num_iters <int>` argument will count all previous checkpoints saved in the directory towards the total number of iterations, and if the existing number of checkpoints exceeds this argment then no new trials will be carried out.

Parallel instances of hyperparameter optimization that share a checkpoint directory will have access to the shared results of hyperparameter optimization trials, allowing them to arrive at the desired total number of iterations collectively more quickly. In this way multiple GPUs or other computing resources can be applied to the search. Each instance of hyperparameter optimization is unaware of parallel trials that have not yet completed. This has several implications when running `n` parallel instances:
* A parallel search will have different information and search different parameters than a single instance sequential search.
* New trials will not consider the parameters in currently running trials, in rare cases leading to duplication.
* Up to `n-1` extra random search iterations may occur above the number specified with `--startup_random_iters`.
* Up to `n-1` extra total trials will be run above the chosen `num_iters`, though each instance will be exposed to at least that number of iterations.
* The last parallel instance to complete is the only one that is aware of all the trials when reporting results.

### Random or Directed Search

As part of the hyperopt search algorithm, the first trial configurations for the model will be randomly spread through the search space. The number of randomized trials can be altered with the argument `--startup_random_iters <int>`. By default, the number or random trials will be half the number of total trials. After this number of trial iterations has been carried out, subsequent trials will use the directed search algorithm to select parameter configurations. This startup count considers the total number of trials in the checkpoint directory rather than the number that has been carried out by an individual instance of hyperparamter optimization instance. Both the random and directed search use a unique trial seed in choosing hyperparameters, this can be specified for the first trial in an optimization instance using `--hyperopt_seed <seed>` and will increment up to the next unused seed for trials afterward.


### Manual Trials

Manual training instances outside of hyperparameter optimization may also be considered in the history of attempted trials. The paths to the save_dirs for these training instances can be specified with `--manual_trial_dirs <list-of-directories>`. These directories must contain the files `test_scores.csv` and `args.json` as generated during training. To work appropriately, these training instances must be consistent with the parameter space being searched in hyperparameter optimization (including the hyperparameter optimization default of ffn_hidden_size being set equal to hidden_size). Manual trials considered with this argument are not added to the checkpoint directory.

## Encode Fingerprint Latent Representation

To load a trained model and encode the fingerprint latent representation of molecules, run `fingerprint.py` and specify:
* `--test_path <path>` Path to the data to predict on.
* A checkpoint by using either:
  * `--checkpoint_dir <dir>` Directory where the model checkpoint is saved (i.e. `--save_dir` during training).
  * `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
* `--preds_path` Path where a CSV file containing the encoded fingerprint vectors will be saved.
* Any other arguments that you would supply for a prediction, such as atom or bond features.

Latent representations of molecules are taken from intermediate stages of the prediction model. This latent representation can be taken at the output of the MPNN (default) or from the last input layer of the FFNN, specified using `--fingerprint_type <MPN or last_FFN>`. Fingerprint encoding uses the same set of arguments as making predictions. If multiple checkpoint files are supplied through `--checkpoint_dir`, then the fingerprint encodings for each of the models will be provided concatenated together as a longer vector.

Example input:
```
chemprop_fingerprint --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints --preds_path tox21_fingerprint.csv
```
or
```
chemprop_fingerprint --test_path data/tox21.csv --checkpoint_path tox21_checkpoints/fold_0/model_0/model.pt --preds_path tox21_fingerprint.csv
```

If installed from source, `chemprop_fingerprint` can be replaced with `python fingerprint.py`.

## Interpreting

It is often helpful to provide explanation of model prediction (i.e., this molecule is toxic because of this substructure). Given a trained model, you can interpret the model prediction using the following command:
```
chemprop_interpret --data_path data/tox21.csv --checkpoint_dir tox21_checkpoints/fold_0/ --property_id 1
```

If installed from source, `chemprop_interpret` can be replaced with `python interpret.py`.

The output will be like the following:
* The first column is a molecule and second column is its predicted property (in this case NR-AR toxicity). 
* The third column is the smallest substructure that made this molecule classified as toxic (which we call rationale). 
* The fourth column is the predicted toxicity of that substructure. 

As shown in the first row, when a molecule is predicted to be non-toxic, we will not provide any rationale for its prediction. 

smiles | NR-AR | rationale | rationale_score
| :---: | :---: | :---: | :---: |
O=\[N+\](\[O-\])c1cc(C(F)(F)F)cc(\[N+\](=O)\[O-\])c1Cl | 0.014 | | | 
CC1(C)O\[C@@H\]2C\[C@H\]3\[C@@H\]4C\[C@H\](F)C5=CC(=O)C=C\[C@\]5(C)\[C@H\]4\[C@@H\](O)C\[C@\]3(C)\[C@\]2(C(=O)CO)O1 | 0.896 | C\[C@\]12C=CC(=O)C=C1\[CH2:1\]C\[CH2:1\]\[CH2:1\]2 | 0.769 |
C\[C@\]12CC\[C@H\]3\[C@@H\](CC\[C@@\]45O\[C@@H\]4C(O)=C(C#N)C\[C@\]35C)\[C@@H\]1CC\[C@@H\]2O | 0.941 | C\[C@\]12C\[CH:1\]=\[CH:1\]\[C@H\]3O\[C@\]31CC\[C@@H\]1\[C@@H\]2CC\[C:1\]\[CH2:1\]1 | 0.808 |
C\[C@\]12C\[C@H\](O)\[C@H\]3\[C@@H\](CCC4=CC(=O)CC\[C@@\]43C)\[C@@H\]1CC\[C@\]2(O)C(=O)COP(=O)(\[O-\])\[O-\] | 0.957 | C1C\[CH2:1\]\[C:1\]\[C@@H\]2\[C@@H\]1\[C@@H\]1CC\[C:1\]\[C:1\]1C\[CH2:1\]2</pre> | 0.532 | 

Chemprop's interpretation script explains model prediction one property at a time. `--property_id 1` tells the script to provide explanation for the first property in the dataset (which is NR-AR). In a multi-task training setting, you will need to change `--property_id` to provide explanation for each property in the dataset.

For computational efficiency, we currently restricted the rationale to have maximum 20 atoms and minimum 8 atoms. You can adjust these constraints through `--max_atoms` and `--min_atoms` argument.

Please note that the interpreting framework is currently only available for models trained on properties of single molecules, that is, multi-molecule models generated via the `--number_of_molecules` command are not supported.

## TensorBoard

During training, TensorBoard logs are automatically saved to the same directory as the model checkpoints. To view TensorBoard logs, first install TensorFlow with `pip install tensorflow`. Then run `tensorboard --logdir=<dir>` where `<dir>` is the path to the checkpoint directory. Then navigate to [http://localhost:6006](http://localhost:6006).

## Results

We compared our model against MolNet by Wu et al. on all of the MolNet datasets for which we could reproduce their splits (all but Bace, Toxcast, and qm7). When there was only one fold provided (scaffold split for BBBP and HIV), we ran our model multiple times and reported average performance. In each case we optimize hyperparameters on separate folds, use rdkit_2d_normalized features when useful, and compare to the best-performing model in MolNet as reported by Wu et al. We did not ensemble our model in these results.

Results on regression datasets (lower is better)

Dataset | Size | Metric | Ours | MolNet Best Model |
| :---: | :---: | :---: | :---: | :---: |
QM8 | 21,786 | MAE | 0.011 ± 0.000 | 0.0143 ± 0.0011 |
QM9 | 133,885 | MAE | 2.666 ± 0.006 | 2.4 ± 1.1 |
ESOL | 1,128 | RMSE | 0.555 ± 0.047 | 0.58 ± 0.03 |
FreeSolv | 642 | RMSE | 1.075 ± 0.054 | 1.15 ± 0.12 |
Lipophilicity | 4,200 | RMSE | 0.555 ± 0.023 | 0.655 ± 0.036 |
PDBbind (full) | 9,880 | RMSE | 1.391 ± 0.012 | 1.25 ± 0 | 
PDBbind (core) | 168 | RMSE | 2.173 ± 0.090 | 1.92 ± 0.07 | 
PDBbind (refined) | 3,040 | RMSE | 1.486 ± 0.026 | 1.38 ± 0 | 

Results on classification datasets (higher is better)

| Dataset | Size | Metric | Ours | MolNet Best Model |
| :---: | :---: | :---: | :---: | :---: |
| PCBA | 437,928 | PRC-AUC | 0.335 ± 0.001 |  0.136 ± 0.004 |
| MUV | 93,087 | PRC-AUC | 0.041 ± 0.007 | 0.184 ± 0.02 |
| HIV | 41,127 | ROC-AUC | 0.776 ± 0.007 | 0.792 ± 0 |
| BBBP | 2,039 | ROC-AUC | 0.737 ± 0.001 | 0.729 ± 0 |
| Tox21 | 7,831 | ROC-AUC | 0.851 ± 0.002 | 0.829 ± 0.006 |
| SIDER | 1,427 | ROC-AUC | 0.676 ± 0.014 | 0.648 ± 0.009 |
| ClinTox | 1,478 | ROC-AUC | 0.864 ± 0.017 | 0.832 ± 0.037 |

Lastly, you can find the code to our original repo at https://github.com/wengong-jin/chemprop and for the Mayr et al. baseline at https://github.com/yangkevin2/lsc_experiments . 
