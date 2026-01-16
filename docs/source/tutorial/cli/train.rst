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
 * :code:`regression-quantile`
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

* **User Specified Splits** Custom splits can be specified in two ways, :code:`--splits-column` and :code:`--splits-file`, examples of which are shown below.

.. code-block::

    chemprop train --splits-column split -i data.csv -t regression

.. list-table:: data.csv
    :widths: 10 10 10
    :header-rows: 1

    * - smiles
      - property
      - split
    * - C
      - 1.0
      - train
    * - CC
      - 2.0
      - train
    * - CCC
      - 3.0
      - test
    * - CCCC
      - 4.0
      - val
    * - CCCCC
      - 5.0
      - val
    * - CCCCCC
      - 6.0
      - test

.. code-block::

    chemprop train --splits-file splits.json -i data.csv -t regression

.. note::
    Use zero-indexing when assigning data indices to different sets. Additionally note that ranges have inclusive ends (ie. [0,1] / "0-1" / "0,1" are equivalent).

.. code-block:: JSON
    :caption: splits.json

    [
        {"train": [0, 1], "val": "2-3", "test": "4,5"},
        {"val": [0, 1], "test": "2-3", "train": "4,5"},
    ]

.. note::
    By default, both random and scaffold split the data into 80% train, 10% validation, and 10% test. This can be changed with :code:`--split-sizes <train_frac> <val_frac> <test_frac>`. The default setting is :code:`--split-sizes 0.8 0.1 0.1`. Both splits also involve a random component that can be seeded with :code:`--data-seed <seed>`. The default setting is :code:`--data-seed 0`.

Other supported splitting methods include :code:`random_with_repeated_smiles`, :code:`kennard_stone`, and :code:`kmeans`.

Replicates
^^^^^^^^^^

Repeat random trials (i.e. replicates) run by specifying :code:`--num-replicates <n>` (default 1, i.e. no replicates).
This is analogous to the 'outer loop' of nested cross validation but at a lower cost, suitable for deep learning applications.

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
 * :code:`--activation <activation_type>` The activation function used in the MPNN and FNN layers. Run :code:`chemprop train -h` to see the full list of activation functions supported via CLI.
 * :code:`--epochs <n>` How many epochs to train over (default 50)
 * :code:`--warmup-epochs <n>`: The number of epochs during which the learning rate is linearly incremented from :code:`init_lr` to :code:`max_lr` (default 2)
 * :code:`--init-lr <n>` Initial learning rate (default 0.0001)
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
 * :code:`dirichlet` Dirichlet


**Multiclass**:

 * :code:`ce` Cross-entropy (default)
 * :code:`multiclass-mcc` Multiclass Matthews correlation coefficient
 * :code:`dirichlet` Dirichlet

**Spectral**:

 * :code:`sid` Spectral information divergence (default)
 * :code:`earthmovers` Earth mover's distance (or first-order Wasserstein distance)
 * :code:`wasserstein` See above.

Evaluation Metrics
------------------

The following evaluation metrics are supported during training:

**Regression**:

 * :code:`rmse` Root mean squared error
 * :code:`mae` Mean absolute error
 * :code:`mse` Mean squared error (default)
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

Pretraining and Transfer Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An existing model, for example from training on a larger, lower quality dataset, can be used for parameter-initialization of a new model by providing a checkpoint of the existing model using :code:`--checkpoint <path>`. :code:`<model_path>`` is the location of checkpoint(s) or model file(s). It can be a path to either a single pretrained model checkpoint (.ckpt) or single pretrained model file (.pt), a directory that contains these files, or a list of path(s) and directory(s).

When training the new model, its architecture **must** resemble that of the old model. Depending on the similarity of the tasks and datasets, as well as the quality of the old model, the new model might require fewer epochs to achieve optimal performance compared to training from scratch.

It is also possible to freeze the weights of a loaded Chemprop model during training, such as for transfer learning applications. To do so, you first need to load a pre-trained model by specifying its checkpoint file using :code:`--checkpoint <path>`. After loading the model, the MPNN weights can be frozen via :code:`--freeze-encoder`. You can control how the weights are frozen in the FFN layers by using :code:`--frzn-ffn-layers <n>` flag, where the :code:`n` is the first n layers are frozen in the FFN layers. By default, :code:`n` is set to 0, meaning all FFN layers are trainable unless specified otherwise.

Finetuning Foundation Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During finetuning one can pretrain a model on an unrelated task and then re-use the learned representation in a new task to improve predictions. This has the effect of improving predictions, particularly on small datasets, by circumventing the need to the model to re-learn the basic facets of molecules representation. 

Unlike Transfer Learning, this does **not** require that the downstream task's FFN has the same architecture as the pretrained model. When finetuning, the Message Passing (depth, hidden size, activation function, etc.) and Aggregation configurations are fixed to be whatever they were during pretraining, but the FNN is initialized from scratch according to the users request and then trained.

Users can access pretrained foundation models by using the :code:`--from-foundation <name>` command line argument. Currently, the following foundation models are available in Chemprop:

 * :code:`CheMeleon` Mordred-descriptor based foundation model pretrained on 1M molecules from PubChem, suitable for many tasks and especially small datasets. See the `CheMeleon GitHub repository <https://github.com/JacksonBurns/chemeleon>`_ for more information.
 * :code:`<your-model>.pt` specify a filepath for a Chemprop model trained via the CLI and the Message Passing will be re-used with a new FFN

The first time a given model is requested it will automatically be downloaded for you and saved to a directory called `.chemprop` in your home directory (except for your own models).

.. _performant-training:

Performant Training
^^^^^^^^^^^^^^^^^^^

By default, graph featurization occurs a single time at the beginning of the training run, and the results are cached for use during each training epoch. This saves time but requires more memory. This behavior can be turned off by specifying :code:`--no-cache`. In either case, graph featurization can be sped up by using more CPU cores, specified via :code:`--num-workers`. This will also convert SMILES strings to :code:`Chem.Mol` objects in parallel and compute any molecule features specified with :code:`--molecule-featurizers` in parallel.

.. note::
  Setting :code:`num_workers` to a value greater than 0 can cause hangs on Windows and MacOS

Training can be further accelerated using a molecular featurizer package called ``cuik-molmaker``. This package is not installed by default, but can be installed using the script ``check_and_install_cuik_molmaker.py``. In order to enable the accelerated featurizer, use the :code:`--use-cuikmolmaker-featurization` flag. This featurizer also performs on-the-fly featurization of molecules and reduces memory usage which is particularly useful for large datasets.


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

Spectra training is different than other datatypes because it considers the predictions of all targets together. Targets for spectra should be provided as the values for the spectrum at a specific position in the spectrum. Spectra predictions are configured to return only positive values and normalize them to sum each spectrum to 1. Spectral prediction are still in beta and will be updated in the future.

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


Extra Datapoint Descriptors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional datapoint descriptors can be concatenated to the learned representation after aggregation. These extra descriptors could be molecule-level features. If you install from source, you can modify the code to load custom descriptors as follows:

1. **Generate features:** If you want to generate molecular features in code, you can write a custom features generator function using the default featurizers in :code:`chemprop/featurizers/`. This also works for custom atom and bond features.
2. **Load features:** Additional descriptors can be provided using :code:`--descriptors-path /path/to/descriptors.npz` where the descriptors are saved as a numpy :code:`.npz` file. This file can be saved using :code:`np.savez("/path/to/descriptors.npz", X_d)`, where :code:`X_d` is a 2D array with a shape of number of datapoints by number of additional descriptors. Note that the descriptors must be in the same order as the SMILES strings in your data file. The extra descriptors are scaled by default. This can be disabled with the option :code:`--no-descriptor-scaling`.


Molecule-Level 2D Features
^^^^^^^^^^^^^^^^^^^^^^^^^^

Chemprop provides several molecule featurizers that automatically calculate molecular features and uses them as extra datapoint descriptors. These are specified using :code:`--molecule-featurizers` followed by one or more of the following:

 * :code:`morgan_binary` binary Morgan fingerprints, radius 2 and 2048 bits
 * :code:`morgan_count` count-based Morgan, radius 2 and 2048 bits
 * :code:`rdkit_2d` RDKit 2D features
 * :code:`v1_rdkit_2d` The RDKit 2D features used in Chemprop v1
 * :code:`v1_rdkit_2d_normalized` The normalized RDKit 2D features used in Chemprop v1

.. note::
   The Morgan fingerprints should not be scaled. Use :code:`--no-descriptor-scaling` to ensure this.

   The RDKit 2D features are not normalized. The :code:`StandardScaler` used in the CLI to normalize is non-optimal for some of the RDKit features. It is recommended to precompute and scale these features outside of the CLI using an appropriate scaler and then provide them using :code:`--descriptors-path` and :code:`--no-descriptor-scaling` as described above.

   In Chemprop v1, :code:`descriptastorus` was used to calculate RDKit 2D features. This package offers normalization of the features, with the normalizations fit to a set of molecules randomly selected from ChEMBL. Several descriptors have been added to :code:`rdkit` recently which are not included in :code:`descriptastorus` including 'AvgIpc', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', and 'SPS'.


Missing Target Values
^^^^^^^^^^^^^^^^^^^^^

When training multitask models (models which predict more than one target simultaneously), sometimes not all target values are known for all molecules in the dataset. Chemprop automatically handles missing entries in the dataset by masking out the respective values in the loss function, so that partial data can be utilized.

The loss function is rescaled according to all non-missing values, and missing values do not contribute to validation or test errors. Training on partial data is therefore possible and encouraged (versus taking out datapoints with missing target entries). No keyword is needed for this behavior, it is the default.


TensorBoard
^^^^^^^^^^^

During training, TensorBoard logs are automatically saved to the output directory under :code:`model_{i}/trainer_logs/version_0/`.
.. To view TensorBoard logs, run :code:`tensorboard --logdir=<dir>` where :code:`<dir>` is the path to the checkpoint directory. Then navigate to `<http://localhost:6006>`_.
