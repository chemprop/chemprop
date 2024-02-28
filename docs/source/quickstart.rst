.. _quickstart:

Quickstart
==========

To get started with Chemprop, first install the package from PyPI::

    pip install chemprop

Let's use the solubility data that comes pre-packaged in the Chemprop directory:

.. code-block:: text

    $ head tests/data/regression.csv
    smiles,logSolubility
    OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O,-0.77
    Cc1occc1C(=O)Nc2ccccc2,-3.3
    CC(C)=CCCC(C)=CC(=O),-2.06
    c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43,-7.87
    c1ccsc1,-1.33
    c2ccc1scnc1c2,-1.5
    Clc1cc(Cl)c(c(Cl)c1)c2c(Cl)cccc2Cl,-7.32
    CC12CCC3C(CCc4cc(O)ccc34)C2CCC1O,-5.03
    ClC4=C(Cl)C5(Cl)C3C1CC(C2OC12)C3C4(Cl)C5(Cl)Cl,-6.29
    ...

Now we're ready to train a simple Chemprop model:

.. code-block:: bash

    chemprop train --data-path tests/data/regression.csv \
        --task-type regression \
        --output-dir train_example

This will train a model on the solubility dataset (``tests/data/regression.csv``) and save the model and training logs in the ``train_example`` directory. You should see some output printed to your terminal that shows the model architecture, number of parameters, and a progress bar for each epoch of training. At the end, you should see something like:

.. code-block:: text
    ───────────────────────────────────────────────────────
       Test metric             DataLoader 0
    ───────────────────────────────────────────────────────
        test/mse             0.7716904154601469
    ───────────────────────────────────────────────────────

With our trained model in hand, we can now use it to predict solubilities of new molecules. In the absence of additional data, for demonstration purposes, let's just test on the same molecules that we trained on:

.. code-block:: bash

    chemprop predict --test-path tests/data/regression.csv \
        --model-path train_example/model_0/model.pt \
        --preds-path train_example/predictions.csv

This should output a file ``train_example/predictions.csv`` containing the predicted log(solubility) values for the molecules contained in ``tests/data/regression.csv``.

.. code-block:: text

    $ head train_example/predictions.csv
    smiles,logSolubility,pred
    OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O,-0.77,...
    Cc1occc1C(=O)Nc2ccccc2,-3.3,...
    CC(C)=CCCC(C)=CC(=O),-2.06,...
    ...

Given that our test data is identical to our training data, it makes sense that the predictions are similar to the ground truth values.

In the rest of this documentation, we'll go into more detail about how to:

* :ref:`Install Chemprop<installation>`
* :ref:`Customize model architecture and task type<train>`
* :ref:`Specify training parameters: split type, learning rate, batch size, loss function, etc.<train>`
* :ref:`Use Chemprop as a Python package<python usage>`
..
     Optimize hyperparameters
    * :ref:`Quantify prediction uncertainty<predict>`

Summary
-------

* Install Chemprop with ``pip install chemprop``
* Train a model with ``chemprop train INPUT --task-type TYPE --output-dir DIR``
* Use a saved model for prediction with ``chemprop predict INPUT --checkpoint-dir DIR --preds-path PATH``

.. _GitHub repository: https://github.com/chemprop/chemprop
..
    .. _FreeSolv dataset: https://pubmed.ncbi.nlm.nih.gov/24928188/