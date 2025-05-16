.. _mol_atom_bond:

Atom and Bond Properties
=========================

In addition to fitting MPNN models to predict molecule-level properties, Chemprop can also fit to atom and bond properties by passing the learned node and edge embeddings from message passing through separate FFNs.

To train a model on atom and bond properties, use the flags :code:`--mol-target-columns`, :code:`--atom-target-columns`, and :code:`--bond-target-columns` instead of :code:`--target-columns`.


Input Data
----------

Target values for atoms and bond can be specified as a list of values, with the order of values matching the order of atoms and bonds in the input SMILES string. For example:

.. code-block::

    smiles,mol_y1,mol_y2,atom_y1,atom_y2,bond_y1,bond_y2
    [H][H],2.016,2.0,"[1, 1]","[1.008, 1.008]",[2],[-2]
    C,16.043,1.0,[6],[12.011],[],[]
    CC#N,41.053,3.0,"[6, 6, 7]","[12.011, 12.011, 14.007]","[12, 13]","[-12, -13]"
    ...

The order of atoms in can be different from the order in the SMILES string if atom maps are used. In this case the :code:`--reorder-atoms` flag should be given. These three datapoints would be equivalent:

.. code-block::

    smiles,atom_y
    [C:1][N:2],"[6, 7]"
    [N:2][C:1],"[6, 7]"
    NC,"[7, 6]"

Reordering the atoms does not change the order of bonds, but the bond targets can also be specified as a list of lists which form an array of shape (n_atoms, n_atoms) where the value at index (i, j) corresponds to the bond between atom i and atom j. For example:

.. code-block::

    smiles,bond_y
    N#CC=N,"[[0, 3, 0, 0], [3, 0, 1, 0], [0, 1, 0, 2], [0, 0, 2, 0]]"
    ...

The model predictions are still returned as a list of values which follow the order of the bonds in the input SMILES string.


Constrained Prediction
----------------------

Some atom and bond properties must sum to a molecule-level value, such as partial charges sum to molecular charge. These constraints are given to :code:`--constraints-path` as a csv file with the following format:

.. code-block::

    constraint_1,constraint_2
    0,12.01
    -1,36.04
    ...

If using constraints, you must indicate which constraint column corresponds to which atom or bond target column. This is done by passing a sequence of strings with the :code:`--constraints-to-targets` flag. The order of the strings matches the order of the constraint columns. The strings must look like 'atom_target_{i}' or 'bond_target_{i}', where i is the index of the atom or bond target column. The index of the atom and bond target columns is determined by the order they were passed using :code:`--atom-target-columns` and :code:`--bond-target-columns`.


Extra Bond Descriptors
---------------------------------

Extra bond descriptors can be used, analogous to the extra atom descriptors. Relevant flags include :code:`--bond-descriptors-path`, :code:`--cal-bond-descriptors-path`, and :code:`--no-bond-descriptor-scaling`.