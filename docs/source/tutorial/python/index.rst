.. _python usage:

Python Module Tutorials
=======================

Chemprop may be used in python scripts, allowing for greater flexibility and control than the CLI. We recommend first looking through some of the worked examples to get an overview of the workflow. Then further details about the creation, customization, and use of Chemprop modules can be found in the following module tutorials:

Data Modules:

* :doc:`data/datapoints`
* :doc:`data/datasets`
* :doc:`data/dataloaders`
* :doc:`data/splitting`

Featurization Modules:

* :doc:`featurizers/atom_featurizers`
* :doc:`featurizers/bond_featurizers`
* :doc:`featurizers/molgraph_molecule_featurizer`
* :doc:`featurizers/molgraph_reaction_featurizer`
* :doc:`featurizers/molecule_featurizers`

Model Modules:

* :doc:`models/basic_mpnn_model`
* :doc:`models/message_passing`
* :doc:`models/aggregation`
* :doc:`models/predictor`
* :doc:`models/multicomponent_mpnn_model`

Other module and workflow tutorials:

* :doc:`activation`
* :doc:`loss_functions`
* :doc:`metrics`
* :doc:`saving_and_loading`
* :doc:`ensembling`
* :doc:`scaling`

.. toctree::
    :maxdepth: 1
    :hidden:

    data/datapoints
    data/datasets
    data/dataloaders
    data/splitting
    featurizers/atom_featurizers
    featurizers/bond_featurizers
    featurizers/molgraph_molecule_featurizer
    featurizers/molgraph_reaction_featurizer
    featurizers/molecule_featurizers
    models/basic_mpnn_model
    models/message_passing
    models/aggregation
    models/predictor
    models/multicomponent_mpnn_model
    activation
    loss_functions
    metrics
    saving_and_loading
    ensembling
    scaling