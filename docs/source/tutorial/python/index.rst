.. _python usage:

Python Usage
============

.. warning:: 
    This page is deprecated. Please see Jupyter notebooks for up-to-date information on how to use Chemprop in Python scripts.


An example of basic Chemprop model training and prediction in Python is given in the :code:`example.py` file. We describe the steps in more detail below:

First, we must import the necessary modules:

.. code-block::
  
  import csv
  import sys

  from lightning import pytorch as pl
  import numpy as np
  from sklearn.model_selection import train_test_split

  from chemprop.v2 import data
  from chemprop.v2 import featurizers
  from chemprop.v2.models import loss, modules, models, metrics

We then define the structure of the neural network by selecting a featurizer, message passing module, aggregation module, and feed-forward network module. We also define the model itself by combining these modules with the :code:`MPNN` class, where we can also specify the metrics to be used during training.

.. code-block::

  featurizer = featurizers.MoleculeMolGraphFeaturizer()
  mp = modules.BondMessageBlock()
  agg = modules.MeanAggregation()
  ffn = modules.RegressionFFN()
  mpnn = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[metrics.RMSEMetric()])

Next, we load the data from a CSV file and split it into training, validation, and test sets.

.. code-block::

  with open(args.input) as fid:
      reader = csv.reader(fid)
      next(reader)
      smis, ys = zip(*[(smi, float(score)) for smi, score in reader])
  ys = np.array(ys).reshape(-1, 1)
  all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

  train_data, val_test_data = train_test_split(all_data, test_size=0.1)
  val_data, test_data = train_test_split(val_test_data, test_size=0.5)

We then create :code:`MoleculeDataset` objects from the data and use them to create PyTorch :code:`DataLoader` objects that can be used to train the model.

.. code-block::

  train_dset = data.MoleculeDataset(train_data, featurizer)
  scaler = train_dset.normalize_targets()
  val_dset = data.MoleculeDataset(val_data, featurizer)
  val_dset.normalize_targets(scaler)
  test_dset = data.MoleculeDataset(test_data, featurizer)
  test_dset.normalize_targets(scaler)

  train_loader = data.build_dataloader(train_dset, num_workers=args.num_workers)
  val_loader = data.build_dataloader(val_dset, num_workers=args.num_workers, shuffle=False)
  test_loader = data.build_dataloader(test_dset, num_workers=args.num_workers, shuffle=False)

Finally, we train the model and evaluate it on the test set.

.. code-block::

  trainer = pl.Trainer(
      logger=False,
      enable_checkpointing=False,
      enable_progress_bar=True,
      accelerator="auto",
      devices=1,
      max_epochs=20,
  )
  trainer.fit(mpnn, train_loader, val_loader)
  results = trainer.test(mpnn, test_loader)
