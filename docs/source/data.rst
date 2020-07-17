Data
====

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
