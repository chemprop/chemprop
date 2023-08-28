.. _quickstart:

Quickstart
==========

To get started with chemprop, first install the package from PyPI::

    pip install chemprop

Next, download a tarball of datasets from from the `GitHub repository`_ and unpack it:

.. code-block:: bash

    wget https://raw.githubusercontent.com/chemprop/chemprop/data.tar.gz
    tar -xvzf data.tar.gz

Let's use the `FreeSolv dataset`_, a collection of experimental and calculated hydration free energies for small molecules, as an example::

    $ head data/freesolv.csv
    smiles,freesolv
    CN(C)C(=O)c1ccc(cc1)OC,-11.01
    CS(=O)(=O)Cl,-4.87
    CC(C)C=C,1.83
    CCc1cnccn1,-5.45
    CCCCCCCO,-4.21
    Cc1cc(cc(c1)O)C,-6.27
    CC(C)C(C)C,2.34
    CCCC(C)(C)O,-3.92
    C[C@@H]1CCCC[C@@H]1C,1.58

Now we're ready to train a simple Chemprop model:

.. code-block:: bash

    chemprop train data/freesolv.csv --dataset-type regression --output-dir freesolv

This will train a model on the FreeSolv dataset (``data/freesolv.csv``) and save the model and training logs in the ``freesolv`` directory. You should see some output printed to your terminal:

.. code-block:: text

    Training for property freesolv
    Splitting data with seed 0
    Training for 100 epochs...
    | MSE: 0.00000, MAE: 0.00000, R^2: 1.00000

With our trained model in hand, we can now use it to predict the solvation free energy of some new molecules:

.. code-block:: bash

    chemprop predict data/freesolv.csv \
        --checkpoint-dir freesolv \
        --preds_path freesolv/predictions.csv

This should output a file ``freesolv/predictions.csv`` containing the predicted solvation free energies for the molecules contained in ``data/freesolv.csv``.

.. code-block:: text

    $ head freesolv/predictions.csv
    smiles,prediction
    CN(C)C(=O)c1ccc(cc1)OC,-11.01
    CS(=O)(=O)Cl,-4.87
    CC(C)C=C,1.83
    ...

Given that our test data is identical to our train data, it makes sense that the predictions are nearly identical to the ground truth values.

In the rest of this documentation, we'll go into more detail about how to:

* Install Chemprop
* Customize model architecture and task type
* Specify training parameters: split type, learning rate, batch size, loss function, etc.
* Quantify prediction uncertainty
* Optimize hyperparameters
* Use Chemprop as a python pacakge

.. note::

    The above list should use hyperlinks to the relevant sections of the documentation.

Summary
-------

* Install chemprop with ``pip install chemprop``
* Train a model with ``chemprop train INPUT dataset-type TYPE --output-dir DIR``
* Use a saved model for prediction with ``chemprop predict INPUT --checkpoint-dir DIR --preds_path PATH``

.. _GitHub repository: https://github.com/chemprop/chemprop
.. _FreeSolv dataset: https://pubmed.ncbi.nlm.nih.gov/24928188/