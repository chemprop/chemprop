.. _tutorial:

Tutorial
========

Data
----

In order to train a model, you must provide training data containing molecules (as SMILES strings) and known target values. Targets can either be real numbers, if performing regression, or binary (i.e. 0s and 1s), if performing classification. Target values which are unknown can be left as blanks.

Our model can either train on a single target ("single tasking") or on multiple targets simultaneously ("multi-tasking").

The data file must be be a **CSV file with a header row**. For example:

.. code-block::

   smiles,NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53
   CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,1,,,0,0,1,0,0,0,0
   CCN1C(=O)NC(c2ccccc2)C1=O,0,0,0,0,0,0,0,,0,,0,0
   ...

By default, it is assumed that the SMILES are in the first column and the targets are in the remaining columns. However, the specific columns containing the SMILES and targets can be specified using the :code:`--smiles_column <column>` and :code:`--target_columns <column_1> <column_2> ...` flags, respectively.

Datasets from `MoleculeNet <http://moleculenet.ai/>`_ and a 450K subset of ChEMBL from `<http://www.bioinf.jku.at/research/lsc/index.html>`_ have been preprocessed and are available in `data.tar.gz <https://github.com/chemprop/chemprop/blob/master/data.tar.gz>`_. To uncompress them, run :code:`tar xvzf data.tar.gz`.

Training
--------

To train a model, run:

.. code-block::

   chemprop_train --data_path <path> --dataset_type <type> --save_dir <dir>

where :code:`<path>` is the path to a CSV file containing a dataset, :code:`<type>` is either "classification" or "regression" depending on the type of the dataset, and :code:`<dir>` is the directory where model checkpoints will be saved.

For example:

.. code-block::

   chemprop_train --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints

A full list of available command-line arguments can be found in :ref:`args`.

If installed from source, :code:`chemprop_train` can be replaced with :code:`python train.py`.

Notes:

* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with :code:`--metric <metric>`.
* :code:`--save_dir` may be left out if you don't want to save model checkpoints.
* :code:`--quiet` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the :code:`save_dir`.


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
   
Predicting
----------

To load a trained model and make predictions, run :code:`predict.py` and specify:

* :code:`--test_path <path>` Path to the data to predict on.
* A checkpoint by using either:

  * :code:`--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. :code:`--save_dir` during training). This will walk the directory, load all :code:`.pt` files it finds, and treat the models as an ensemble.
  * :code:`--checkpoint_path <path>` Path to a model checkpoint file (:code:`.pt` file).

* :code:`--preds_path` Path where a CSV file containing the predictions will be saved.

For example:

.. code-block::

   chemprop_predict --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints --preds_path tox21_preds.csv

or

.. code-block::

   chemprop_predict --test_path data/tox21.csv --checkpoint_path tox21_checkpoints/fold_0/model_0/model.pt --preds_path tox21_preds.csv

If installed from source, :code:`chemprop_predict` can be replaced with :code:`python predict.py`.

Interpreting
^^^^^^^^^^^^

It is often helpful to provide explanation of model prediction (i.e., this molecule is toxic because of this substructure). Given a trained model, you can interpret the model prediction using the following command:

.. code-block::

   chemprop_interpret --data_path data/tox21.csv --checkpoint_dir tox21_checkpoints/fold_0/ --property_id 1

If installed from source, :code:`chemprop_interpret` can be replaced with :code:`python interpret.py`.

The output will be like the following:

* The first column is a molecule and second column is its predicted property (in this case NR-AR toxicity).
* The third column is the smallest substructure that made this molecule classified as toxic (which we call rationale).
* The fourth column is the predicted toxicity of that substructure.

As shown in the first row, when a molecule is predicted to be non-toxic, we will not provide any rationale for its prediction.

.. csv-table::
   :header: "smiles", "NR-AR", "rationale", "rationale_score"
   :widths: 20, 10, 20, 10

   "O=[N+]([O-])c1cc(C(F)(F)F)cc([N+](=O)[O-])c1Cl", "0.014", "", ""
   "CC1(C)O[C@@H]2C[C@H]3[C@@H]4C[C@H](F)C5=CC(=O)C=C[C@]5(C)[C@H]4[C@@H](O)C[C@]3(C)[C@]2(C(=O)CO)O1", "0.896", "C[C@]12C=CC(=O)C=C1[CH2:1]C[CH2:1][CH2:1]2", "0.769"
   "C[C@]12CC[C@H]3[C@@H](CC[C@@]45O[C@@H]4C(O)=C(C#N)C[C@]35C)[C@@H]1CC[C@@H]2O", "0.941", "C[C@]12C[CH:1]=[CH:1][C@H]3O[C@]31CC[C@@H]1[C@@H]2CC[C:1][CH2:1]1", "0.808"
   "C[C@]12C[C@H](O)[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)[C@@H]1CC[C@]2(O)C(=O)COP(=O)([O-])[O-]", "0.957", "C1C[CH2:1][C:1][C@@H]2[C@@H]1[C@@H]1CC[C:1][C:1]1C[CH2:1]2", "0.532"

Chemprop's interpretation script explains model prediction one property at a time. :code:`--property_id 1` tells the script to provide explanation for the first property in the dataset (which is NR-AR). In a multi-task training setting, you will need to change :code:`--property_id` to provide explanation for each property in the dataset.

For computational efficiency, we currently restricted the rationale to have maximum 20 atoms and minimum 8 atoms. You can adjust these constraints through :code:`--max_atoms` and :code:`--min_atoms` argument.

Please note that the interpreting framework is currently only available for models trained on properties of single molecules, that is, multi-molecule models generated via the :code:`--number_of_molecules` command are not supported.

TensorBoard
^^^^^^^^^^^

During training, TensorBoard logs are automatically saved to the same directory as the model checkpoints. To view TensorBoard logs, run :code:`tensorboard --logdir=<dir>` where :code:`<dir>` is the path to the checkpoint directory. Then navigate to `<http://localhost:6006>`_.

Web Interface
-------------

For those less familiar with the command line, Chemprop also includes a web interface which allows for basic training and predicting. See :ref:`web` for more details.
