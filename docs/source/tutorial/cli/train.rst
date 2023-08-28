.. _train:

Training a Chemprop Model
=========================

To train a model, run:

.. code-block::

   chemprop train INPUT_PATH --dataset_type <type> --save_dir <dir>

where ``INPUT_PATH`` is the path to a CSV file containing a dataset, ``<type>`` is either "classification" or "regression" depending on the type of the dataset, and ``<dir>`` is the directory where model checkpoints will be saved.

For example:

.. code-block::

    chemprop train data/tox21.csv \
        --dataset_type classification \
        --save_dir tox21_checkpoints

A full list of available command-line arguments can be found in :ref:`args`.


Input Data
----------

In order to train a model, you must provide training data containing molecules (as SMILES strings) and known target values. Targets can either be real numbers, if performing regression, or binary (i.e. 0s and 1s), if performing classification. Target values which are unknown can be left as blanks. A model can be trained as either single- or multi-task.

The data file must be be a **CSV file with a header row**. For example:

.. code-block::

   smiles,NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53
   CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,1,,,0,0,1,0,0,0,0
   CCN1C(=O)NC(c2ccccc2)C1=O,0,0,0,0,0,0,0,,0,,0,0
   ...

By default, it is assumed that the SMILES are in the first column and the targets are in the remaining columns. However, the specific columns containing the SMILES and targets can be specified using the :code:`--smiles_column <column>` and :code:`--target_columns <column_1> <column_2> ...` flags, respectively.

Datasets from `MoleculeNet <http://moleculenet.ai/>`_ and a 450K subset of ChEMBL from `<http://www.bioinf.jku.at/research/lsc/index.html>`_ have been preprocessed and are available in `data.tar.gz <https://github.com/chemprop/chemprop/blob/master/data.tar.gz>`_. To uncompress them, run :code:`tar xvzf data.tar.gz`.

Notes:

* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with ``--metric METRIC``.
* ``--save_dir`` may be left out if you don't want to save model checkpoints.
* ``--quiet`` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the ``save_dir``.


Train/Validation/Test Splits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our code supports several methods of splitting data into train, validation, and test sets.

**Random:** By default, the data will be split randomly into train, validation, and test sets.

**Scaffold:** Alternatively, the data can be split by molecular scaffold so that the same scaffold never appears in more than one split. This can be specified by adding :code:`--split_type scaffold_balanced`.

**Separate val/test:** If you have separate data files you would like to use as the validation or test set, you can specify them with :code:`--separate_val_path <val_path>` and/or :code:`--separate_test_path <test_path>`.

Note: By default, both random and scaffold split the data into 80% train, 10% validation, and 10% test. This can be changed with :code:`--split_sizes <train_frac> <val_frac> <test_frac>`. For example, the default setting is :code:`--split_sizes 0.8 0.1 0.1`. Both also involve a random component and can be seeded with :code:`--seed <seed>`. The default setting is :code:`--seed 0`.

Cross validation
^^^^^^^^^^^^^^^^

k-fold cross-validation can be run by specifying :code:`--num_folds <k>`. The default is :code:`--num_folds 1`.

Ensembling
^^^^^^^^^^

To train an ensemble, specify the number of models in the ensemble with :code:`--ensemble_size <n>`. The default is :code:`--ensemble_size 1`.

Hyperparameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although the default message passing architecture works quite well on a variety of datasets, optimizing the hyperparameters for a particular dataset often leads to marked improvement in predictive performance. We have automated hyperparameter optimization via Bayesian optimization (using the `hyperopt <https://github.com/hyperopt/hyperopt>`_ package), which will find the optimal hidden size, depth, dropout, and number of feed-forward layers for our model. Optimization can be run as follows:

.. code-block::

   chemprop_hyperopt --data_path <data_path> --dataset_type <type> --num_iters <n> --config_save_path <config_path>


where :code:`<n>` is the number of hyperparameter settings to try and :code:`<config_path>` is the path to a :code:`.json` file where the optimal hyperparameters will be saved.

If installed from source, :code:`chemprop_hyperopt` can be replaced with :code:`python hyperparameter_optimization.py`.

Once hyperparameter optimization is complete, the optimal hyperparameters can be applied during training by specifying the config path as follows:

.. code-block::

   chemprop_train --data_path <data_path> --dataset_type <type> --config_path <config_path>

Note that the hyperparameter optimization script sees all the data given to it. The intended use is to run the hyperparameter optimization script on a dataset with the eventual test set held out. If you need to optimize hyperparameters separately for several different cross validation splits, you should e.g. set up a bash script to run hyperparameter_optimization.py separately on each split's training and validation data with test held out.

Additional Features
^^^^^^^^^^^^^^^^^^^

While the model works very well on its own, especially after hyperparameter optimization, we have seen that additional features can further improve performance on certain datasets. The additional features can be added at the atom-, bond, or molecule-level. Molecule-level features can be either automatically generated by RDKit or custom features provided by the user.

Molecule-Level RDKit 2D Features
""""""""""""""""""""""""""""""""

As a starting point, we recommend using pre-normalized RDKit features by using the :code:`--features_generator rdkit_2d_normalized --no_features_scaling` flags. In general, we recommend NOT using the :code:`--no_features_scaling` flag (i.e. allow the code to automatically perform feature scaling), but in the case of :code:`rdkit_2d_normalized`, those features have been pre-normalized and don't require further scaling.

The full list of available features for :code:`--features_generator` is as follows.

:code:`morgan` is binary Morgan fingerprints, radius 2 and 2048 bits.
:code:`morgan_count` is count-based Morgan, radius 2 and 2048 bits.
:code:`rdkit_2d` is an unnormalized version of 200 assorted rdkit descriptors. Full list can be found at the bottom of our paper: `<https://arxiv.org/pdf/1904.01561.pdf>`_
:code:`rdkit_2d_normalized` is the CDF-normalized version of the 200 rdkit descriptors.

Molecule-Level Custom Features
""""""""""""""""""""""""""""""

If you install from source, you can modify the code to load custom features as follows:

1. **Generate features:** If you want to generate features in code, you can write a custom features generator function in :code:`chemprop/features/features_generators.py`. Scroll down to the bottom of that file to see a features generator code template.
2. **Load features:** If you have features saved as a numpy :code:`.npy` file or as a :code:`.csv` file, you can load the features by using :code:`--features_path /path/to/features`. Note that the features must be in the same order as the SMILES strings in your data file. Also note that :code:`.csv` files must have a header row and the features should be comma-separated with one line per molecule.

Atom-Level Features
"""""""""""""""""""

Similar to the additional molecular features described above, you can also provide additional atomic features via :code:`--atom_descriptors_path /path/to/features` with valid file formats:

* :code:`.npz` file, where descriptors are saved as 2D array for each molecule in the exact same order as the SMILES strings in your data file.
* :code:`.pkl` / :code:`.pckl` / :code:`.pickle` containing a pandas dataframe with smiles as index and numpy array of descriptors as columns.
* :code:`.sdf` containing all mol blocks with descriptors as entries.

The order of the descriptors for each atom per molecule must match the ordering of atoms in the RDKit molecule object. Further information on supplying atomic descriptors can be found `here <https://github.com/chemprop/chemprop/releases/tag/v1.1.0>`_.

Users must select in which way atom descriptors are used. The command line option :code:`--atom_descriptors descriptor` concatenates the new features to the embedded atomic features after the D-MPNN with an additional linear layer. The option :code:`--atom_descriptors feature` concatenates the features to each atomic feature vector before the D-MPNN, so that they are used during message-passing. Alternatively, the user can overwrite the default atom features with the custom features using the option :code:`--overwrite_default_atom_features`.

Similar to the molecule-level features, the atom-level descriptors and features are scaled by default. This can be disabled with the option :code:`--no_atom_descriptor_scaling`

Bond-Level Features
"""""""""""""""""""

Bond-level features can be provided in the same format as the atom-level features, using the option :code:`--bond_features_path /path/to/features`. The order of the features for each molecule must match the bond ordering in the RDKit molecule object.

The bond-level features are concatenated with the bond feature vectors before the D-MPNN, such that they are used during message-passing. Alternatively, the user can overwrite the default bond features with the custom features using the option :code:`--overwrite_default_bond_features`.

Similar to molecule-, and atom-level features, the bond-level features are scaled by default. This can be disabled with the option :code:`--no_bond_features_scaling`.

Reaction
^^^^^^^^

As an alternative to molecule SMILES, Chemprop can also process atom-mapped reaction SMILES (see `Daylight manual <https://www.daylight.com/meetings/summerschool01/course/basics/smirks.html>`_ for details on reaction SMILES), which consist of three parts denoting reactants, agents and products, separated by ">". Use the option :code:`--reaction` to enable the input of reactions, which transforms the reactants and products of each reaction to the corresponding condensed graph of reaction and changes the initial atom and bond features to hold information from both the reactant and product (option :code:`--reaction_mode reac_prod`), or from the reactant and the difference upon reaction (option :code:`--reaction_mode reac_diff`, default) or from the product and the difference upon reaction (option :code:`--reaction_mode prod_diff`). In reaction mode, Chemprop thus concatenates information to each atomic and bond feature vector, for example, with option :code:`--reaction_mode reac_prod`, each atomic feature vector holds information on the state of the atom in the reactant (similar to default Chemprop), and concatenates information on the state of the atom in the product, so that the size of the D-MPNN increases slightly. Agents are discarded. Functions incompatible with a reaction as input (scaffold splitting and feature generation) are carried out on the reactants only. If the atom-mapped reaction SMILES contain mapped hydrogens, enable explicit hydrogens via :code:`--explicit_h`. Example of an atom-mapped reaction SMILES denoting the reaction of methanol to formaldehyde without hydrogens: :code:`[CH3:1][OH:2]>>[CH2:1]=[O:2]` and with hydrogens: :code:`[C:1]([H:3])([H:4])([H:5])[O:2][H:6]>>[C:1]([H:3])([H:4])=[O:2].[H:5][H:6]`. The reactions do not need to be balanced and can thus contain unmapped parts, for example leaving groups, if necessary.
For further details and benchmarking, as well as a citable reference, please see `DOI 10.33774/chemrxiv-2021-frfhz <https://doi.org/10.33774/chemrxiv-2021-frfhz>`_.

Pretraining
^^^^^^^^^^^

An existing model, for example from training on a larger, lower quality dataset, can be used for parameter-initialization of a new model by providing a checkpoint of the existing model using either:

 * :code:`--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. :code:`--save_dir` during training of the old model). This will walk the directory, and load all :code:`.pt` files it finds.
 * :code:`--checkpoint_path <path>` Path to a model checkpoint file (:code:`.pt` file).

when training the new model. The model architecture of the new model should resemble the architecture of the old model - otherwise some or all parameters might not be loaded correctly. Please note that the old model is only used to initialize the parameters of the new model, but all parameters remain trainable (no frozen layers). Depending on the quality of the old model, the new model might only need a few epochs to train.

Missing target values
^^^^^^^^^^^^^^^^^^^^^

When training multitask models (models which predict more than one target simultaneously), sometimes not all target values are known for all molecules in the dataset. Chemprop automatically handles missing entries in the dataset by masking out the respective values in the loss function, so that partial data can be utilized, too. The loss function is rescaled according to all non-missing values, and missing values furthermore do not contribute to validation or test errors. Training on partial data is therefore possible and encouraged (versus taking out datapoints with missing target entries). No keyword is needed for this behavior, it is the default.

In contrast, when using :code:`sklearn_train.py` (a utility script provided within Chemprop that trains standard models such as random forests on Morgan fingerprints via the python package scikit-learn), multi-task models cannot be trained on datasets with partially missing targets. However, one can instead train individual models for each task (via the argument :code:`--single_task`), where missing values are automatically removed from the dataset. Thus, the training still makes use of all non-missing values, but by training individual models for each task, instead of one model with multiple output values. This restriction only applies to sklearn models (via  :code:`sklearn_train` or :code:`python sklearn_train.py`), but NOT to default Chemprop models via :code:`chemprop_train` or :code:`python train.py`.

TensorBoard
^^^^^^^^^^^

During training, TensorBoard logs are automatically saved to the same directory as the model checkpoints. To view TensorBoard logs, run :code:`tensorboard --logdir=<dir>` where :code:`<dir>` is the path to the checkpoint directory. Then navigate to `<http://localhost:6006>`_.
