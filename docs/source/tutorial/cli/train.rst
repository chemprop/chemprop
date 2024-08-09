.. _train:

Training
=========================

To train a model, run:

.. code-block::
   
    chemprop train --data-path <input_path> --task-type <task> --output-dir <dir>

where ``<input_path>`` is the path to a CSV file containing a dataset, ``<task>`` is the type of modeling task, and ``<dir>`` is the directory where model checkpoints will be saved.

For example:

.. code-block::

    chemprop train --data-path tests/data/regression.csv \
        --task-type regression \
        --output-dir solubility_checkpoints

The following modeling tasks are supported:

 * :code:`regression`
 * :code:`regression-mve`
 * :code:`regression-evidential`
 * :code:`classification`
 * :code:`classification-dirichlet`
 * :code:`multiclass`
 * :code:`multiclass-dirichlet`
 * :code:`spectral`

A full list of available command-line arguments can be found in :ref:`cmd`.


Input Data
----------

In order to train a model, you must provide training data containing molecules (as SMILES strings) and known target values. Targets can either be real numbers, if performing regression, or binary (i.e. 0s and 1s), if performing classification. Target values which are unknown can be left as blanks. A model can be trained as either single- or multi-task.

The data file must be be a **CSV file with a header row**. For example:

.. code-block::

    smiles,NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53
    CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,1,,,0,0,1,0,0,0,0
    CCN1C(=O)NC(c2ccccc2)C1=O,0,0,0,0,0,0,0,,0,,0,0
    ...

By default, it is assumed that the SMILES are in the first column and the targets are in the remaining columns. However, the specific columns containing the SMILES and targets can be specified using the :code:`--smiles-columns <column>` and :code:`--target-columns <column_1> <column_2> ...` flags, respectively. To simultaneously train multiple molecules (such as a solute and a solvent), supply two column headers in :code:`--smiles-columns <columns>`.

.. _train_validation_test_splits:

Train/Validation/Test Splits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our code supports several methods of splitting data into train, validation, and test sets.

* **Random:** By default, the data will be split randomly into train, validation, and test sets.

* **Scaffold:** Alternatively, the data can be split by molecular scaffold so that the same scaffold never appears in more than one split. This can be specified by adding :code:`--split-type scaffold_balanced`.

* **User Specified Splits** The ability to specify your own split indices will be added soon.

*Note*: By default, both random and scaffold split the data into 80% train, 10% validation, and 10% test. This can be changed with :code:`--split-sizes <train_frac> <val_frac> <test_frac>`. The default setting is :code:`--split-sizes 0.8 0.1 0.1`. Both splits also involve a random component that can be seeded with :code:`--data-seed <seed>`. The default setting is :code:`--data-seed 0`.

Other supported splitting methods include :code:`cv`, :code:`cv_no_val`, :code:`random_with_repeated_smiles`, :code:`kennard_stone`, and :code:`kmeans`.

Cross Validation
^^^^^^^^^^^^^^^^

k-fold cross-validation can be run by specifying :code:`--num-folds <k>` (default 1, i.e. no cross-validation).

Ensembling
^^^^^^^^^^

To train an ensemble, specify the number of models in the ensemble with :code:`--ensemble-size <n>` (default 1).

Hyperparameters
---------------

Model performance is often highly dependent on the hyperparameters used. Below is a list of common hyperparameters (see :ref:`cmd` for a full list):

 * :code:`--batch-size` Batch size (default 64)
 * :code:`--message-hidden-dim <n>` Hidden dimension of the messages in the MPNN (default 300)
 * :code:`--depth <n>` Number of message-passing steps (default 3)
 * :code:`--dropout <n>` Dropout probability in the MPNN & FFN layers (default 0)
 * :code:`--activation <activation_type>` The activation function used in the MPNN and FNN layers. Options include :code:`relu`, :code:`leakyrelu`, :code:`prelu`, :code:`tanh`, :code:`selu`, and :code:`elu`. (default :code:`relu`)
 * :code:`--epochs <n>` How many epochs to train over (default 50)
 * :code:`--warmup-epochs <n>`: The number of epochs during which the learning rate is linearly incremented from :code:`init_lr` to :code:`max_lr` (default 2)
 * :code:`--init_lr <n>` Initial learning rate (default 0.0001)
 * :code:`--max-lr <n>` Maximum learning rate (default 0.001)
 * :code:`--final-lr <n>` Final learning rate (default 0.0001)


Loss Functions
--------------

The loss function can be specified using the :code:`--loss-function <function>` keyword, where `<function>` is one of the following:

**Regression**:

 * :code:`mse` Mean squared error (default)
 * :code:`bounded-mse` Bounded mean squared error
 * :code:`mve` Mean-variance estimation
 * :code:`evidential` Evidential; if used, :code:`--evidential-regularization` can be specified to modify the regularization, and :code:`--eps` to modify epsilon.

**Classification**:

 * :code:`bce` Binary cross-entropy (default)
 * :code:`binary-mcc` Binary Matthews correlation coefficient
 * :code:`binary-dirichlet` Binary Dirichlet 


**Multiclass**:

 * :code:`ce` Cross-entropy (default)
 * :code:`multiclass-mcc` Multiclass Matthews correlation coefficient 
 * :code:`multiclass-dirichlet` Multiclass Dirichlet

**Spectral**:

 * :code:`sid` Spectral information divergence (default)
 * :code:`earthmovers` Earth mover's distance (or first-order Wasserstein distance)
 * :code:`wasserstein` See above.

Evaluation Metrics
------------------

The following evaluation metrics are supported during training:

**Regression**:

 * :code:`rmse` Root mean squared error (default)
 * :code:`mae` Mean absolute error
 * :code:`mse` Mean squared error
 * :code:`bounded-mae` Bounded mean absolute error
 * :code:`bounded-mse` Bounded mean squared error
 * :code:`bounded-rmse` Bounded root mean squared error
 * :code:`r2` R squared metric 

**Classification**:

 * :code:`roc` Receiver operating characteristic (default)
 * :code:`prc` Precision-recall curve
 * :code:`accuracy` Accuracy
 * :code:`f1` F1 score
 * :code:`bce` Binary cross-entropy
 * :code:`binary-mcc` Binary Matthews correlation coefficient

**Multiclass**:

 * :code:`ce` Cross-entropy (default)
 * :code:`multiclass-mcc` Multiclass Matthews correlation coefficient 

**Spectral**:

 * :code:`sid` Spectral information divergence (default)
 * :code:`wasserstein` Earth mover's distance (or first-order Wasserstein distance)


Advanced Training Methods
-------------------------

Pretraining
^^^^^^^^^^^

.. An existing model, for example from training on a larger, lower quality dataset, can be used for parameter-initialization of a new model by providing a checkpoint of the existing model using either:

..  * :code:`--checkpoint-dir <dir>` Directory where the model checkpoint(s) are saved (i.e. :code:`--save_dir` during training of the old model). This will walk the directory, and load all :code:`.pt` files it finds.
..  * :code:`--checkpoint-path <path>` Path to a model checkpoint file (:code:`.pt` file).
.. when training the new model. The model architecture of the new model should resemble the architecture of the old model - otherwise some or all parameters might not be loaded correctly. Please note that the old model is only used to initialize the parameters of the new model, but all parameters remain trainable (no frozen layers). Depending on the quality of the old model, the new model might only need a few epochs to train.

It is possible to freeze the weights of a loaded model during training, such as for transfer learning applications. To do so, specify :code:`--model-frzn <path>` where :code:`<path>` refers to a model's checkpoint file that will be used to overwrite and freeze the model weights. The following flags may be used:

 * :code:`--frzn-ffn-layers <n>` Overwrites weights for the first n layers of the FFN from the checkpoint (default 0)  

.. _train-on-reactions:

Training on Reactions
^^^^^^^^^^^^^^^^^^^^^

Chemprop can also process atom-mapped reaction SMILES (see `Daylight manual <https://www.daylight.com/meetings/summerschool01/course/basics/smirks.html>`_ for details), which consist of three parts denoting reactants, agents, and products, each separated by ">". For example, an atom-mapped reaction SMILES denoting the reaction of methanol to formaldehyde without hydrogens: :code:`[CH3:1][OH:2]>>[CH2:1]=[O:2]` and with hydrogens: :code:`[C:1]([H:3])([H:4])([H:5])[O:2][H:6]>>[C:1]([H:3])([H:4])=[O:2].[H:5][H:6]`. The reactions do not need to be balanced and can thus contain unmapped parts, for example leaving groups, if necessary.

Specify columns in the input file with reaction SMILES using the option :code:`--reaction-columns` to enable this, which transforms the reactants and products to the corresponding condensed graph of reaction, and changes the initial atom and bond features depending on the argument provided to :code:`--rxn-mode <feature_type>`:

 * :code:`reac_diff` Featurize with the reactant and the difference upon reaction (default)
 * :code:`reac_prod` Featurize with both the reactant and product
 * :code:`prod_diff` Featurize with the product and the difference upon reaction

Each of these arguments can be modified to balance imbalanced reactions by appending :code:`_balance`, e.g. :code:`reac_diff_balance`. 

In reaction mode, Chemprop concatenates information to each atomic and bond feature vector. For example, using :code:`--reaction-mode reac_prod`, each atomic feature vector holds information on the state of the atom in the reactant (similar to default Chemprop), and concatenates information on the state of the atom in the product. Agents are featurized with but not connected to the reactants. Functions incompatible with a reaction as input (scaffold splitting and feature generation) are carried out on the reactants only. 

If the atom-mapped reaction SMILES contain mapped hydrogens, enable explicit hydrogens via :code:`--keep-h`.

For further details and benchmarking, as well as a citable reference, please see `DOI 10.1021/acs.jcim.1c00975 <https://doi.org/10.1021/acs.jcim.1c00975>`_.


Training Reactions with Molecules (e.g. Solvents, Reagents)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both reaction and molecule SMILES can be associated with a target (e.g. a reaction rate in a solvent). To do so, use both :code:`--smiles-columns` and :code:`--reaction-columns`.

.. Chemprop allows differently-sized MPNNs to be used for each reaction and solvent/molecule encoding. The following commands can be used to specify the solvent/molecule MPNN size if :code:`--reaction-solvent` is specified:

..  * :code:`--bias-solvent` Whether to add bias to the linear layers of the solvent/molecule (default :code:`false`)
..  * :code:`--hidden-size-solvent <n>` The dimensionality of the hidden layers for the solvent/molecule (default 300)
..  * :code:`--depth-solvent <n>` The number of message passing steps for the solvent/molecule (default 3)

The reaction and molecule SMILES columns can be ordered in any way. However, the same column ordering as used in the training must be used for the prediction. For more information on atom-mapped reaction SMILES, please refer to :ref:`train-on-reactions`.


Training on Spectra
^^^^^^^^^^^^^^^^^^^

Spectra training is different than other datatypes because it considers the predictions of all targets together. Targets for spectra should be provided as the values for the spectrum at a specific position in the spectrum. Spectra predictions are configured to return only positive values and normalize them to sum each spectrum to 1. 
.. Activation to enforce positivity is an exponential function by default but can also be set as a Softplus function, according to the argument :code:`--spectral-activation <exp or softplus>`. Value positivity is enforced on input targets as well using a floor value that replaces negative or smaller target values with the floor value, customizable with the argument :code:`--spectra_target_floor <float>` (default 1e-8).

.. In absorption spectra, sometimes the phase of collection will create regions in the spectrum where data collection or prediction would be unreliable. To exclude these regions, include paths to phase features for your data (:code:`--phase-features-path <path>`) and a mask indicating the spectrum regions that are supported (:code:`--spectra-phase-mask-path <path>`). The format for the mask file is a .csv file with columns for the spectrum positions and rows for the phases, with column and row labels in the same order as they appear in the targets and features files.


Additional Features
-------------------

While the model works very well on its own, especially after hyperparameter optimization, additional features and descriptors may further improve performance on certain datasets. Features are used before message passing while descriptors are used after message passing. The additional features/descriptors can be added at the atom-, bond, or molecule-level. Molecule-level features can be either automatically generated by RDKit or custom features provided by the user and are concatenated to the learned descriptors generated by Chemprop during message passing (i.e. used as extra descriptors).


Atom-Level Features/Descriptors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can provide additional atom features via :code:`--atom-features-path /path/to/atom/features.npz` as a numpy :code:`.npz` file. This command concatenates the features to each atomic feature vector before the D-MPNN, so that they are used during message-passing. This file can be saved using :code:`np.savez("atom_features.npz", *V_fs)`, where :code:`V_fs` is a list containing the atom features :code:`V_f` for each molecule, where :code:`V_f` is a 2D array with a shape of number of atoms by number of atom features in the exact same order as the SMILES strings in your data file.

Similarly, you can provide additional atom descriptors via :code:`--atom-descriptors-path /path/to/atom/descriptors.npz` as a numpy :code:`.npz` file. This command concatenates the new features to the embedded atomic features after the D-MPNN with an additional linear layer. This file can be saved using :code:`np.savez("atom_descriptors.npz", *V_ds)`, where :code:`V_ds` has the same format as :code:`V_fs` above.

The order of the atom features and atom descriptors for each atom per molecule must match the ordering of atoms in the RDKit molecule object. 

The atom-level features and descriptors are scaled by default. This can be disabled with the option :code:`--no-atom-feature-scaling` or :code:`--no-atom-descriptor-scaling`.


Bond-Level Features
^^^^^^^^^^^^^^^^^^^

Bond-level features can be provided using the option :code:`--bond-features-path /path/to/bond/features.npz`. as a numpy :code:`.npz` file. This command concatenates the features to each bond feature vector before the D-MPNN, so that they are used during message-passing. This file can be saved using :code:`np.savez("bond_features.npz", *E_fs)`, where :code:`E_fs` is a list containing the bond features :code:`E_f` for each molecule, where :code:`E_f` is a 2D array with a shape of number of bonds by number of bond features in the exact same order as the SMILES strings in your data file.

The order of the bond features for each molecule must match the bond ordering in the RDKit molecule object.

Note that bond descriptors are not currently supported because the post message passing readout function aggregates atom descriptors. 

The bond-level features are scaled by default. This can be disabled with the option :code:`--no-bond-features-scaling`.


Extra Descriptors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional descriptors can be concatenated to the learned representation after aggregation. These could be molecule features, for example. If you install from source, you can modify the code to load custom descriptors as follows:

1. **Generate features:** If you want to generate molecule features in code, you can write a custom features generator function using the default featurizers in :code:`chemprop/featurizers/`. This also works for custom atom and bond features. 
2. **Load features:** Additional descriptors can be provided using :code:`--descriptors-path /path/to/descriptors.npz` as a numpy :code:`.npz` file. This file can be saved using :code:`np.savez("/path/to/descriptors.npz", X_d)`, where :code:`X_d` is a 2D array with a shape of number of datapoints by number of additional descriptors. Note that the descriptors must be in the same order as the SMILES strings in your data file. The extra descriptors are scaled by default. This can be disabled with the option :code:`--no-descriptor-scaling`.


Molecule-Level 2D Features
^^^^^^^^^^^^^^^^^^^^^^^^^^

Morgan fingerprints can be generated as molecular 2D features using :code:`--molecule-featurizers`:

* :code:`morgan_binary` binary Morgan fingerprints, radius 2 and 2048 bits.
* :code:`morgan_count` count-based Morgan, radius 2 and 2048 bits.


Missing Target Values
^^^^^^^^^^^^^^^^^^^^^

When training multitask models (models which predict more than one target simultaneously), sometimes not all target values are known for all molecules in the dataset. Chemprop automatically handles missing entries in the dataset by masking out the respective values in the loss function, so that partial data can be utilized. 

The loss function is rescaled according to all non-missing values, and missing values do not contribute to validation or test errors. Training on partial data is therefore possible and encouraged (versus taking out datapoints with missing target entries). No keyword is needed for this behavior, it is the default.


TensorBoard
^^^^^^^^^^^

During training, TensorBoard logs are automatically saved to the output directory under :code:`model_{i}/trainer_logs/version_0/`. 
.. To view TensorBoard logs, run :code:`tensorboard --logdir=<dir>` where :code:`<dir>` is the path to the checkpoint directory. Then navigate to `<http://localhost:6006>`_.
