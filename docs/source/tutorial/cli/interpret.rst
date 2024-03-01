.. _interpret:

Interpreting
============

.. warning:: 
    This page is under construction.

.. 
   It is often helpful to provide explanation of model prediction (i.e., this molecule is toxic because of this substructure). Given a trained model, you can interpret the model prediction using the following command:

   .. code-block::

      chemprop interpret --data_path data/tox21.csv --checkpoint_dir tox21_checkpoints/fold_0/ --property_id 1

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
