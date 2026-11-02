.. _convert:

Conversion
----------

To convert a trained model from any version of Chemprop to the latest format, run ``chemprop convert`` and specify:

 * :code:`--input-path <path>` Path of the model file (``.pt``) or checkpoint (``.ckpt``) to convert.
 * :code:`--output-path <path>` Path where the converted model will be saved. If unspecified, this will default to ``<CURRENT_DIRECTORY/STEM_OF_INPUT>_latest.pt`` (or ``.ckpt`` for checkpoints).

This applies each of the conversions below as needed. The converted model file can be loaded with :code:`--weights-only true`. Models converted from v1 must be run with the v1 featurizer via :code:`--multi-hot-atom-featurizer-mode v1`.

.. warning::
    Converting a model file saved before Chemprop v2.4 requires loading it with :code:`--weights-only false`, which can execute arbitrary code, so only convert model files you trust. If :code:`--weights-only` is not specified, ``false`` is used with a warning that the default will change to ``true`` in a future release.

The individual conversions can also be run with :code:`--conversion <conversion>`:

 * :code:`v1_to_v2` converts a trained model from Chemprop v1 to v2. The output defaults to ``<CURRENT_DIRECTORY/STEM_OF_INPUT>_v2.pt``.
 * :code:`v2_0_to_v2_1` converts a trained model from Chemprop v2.0.x to v2.1.y (or newer). The output defaults to ``<CURRENT_DIRECTORY/STEM_OF_INPUT>_v2_1.pt``.
 * :code:`v2_1_to_v2_4` converts a trained model from Chemprop v2.1 or newer to the format that can be loaded with :code:`--weights-only true`. The output defaults to ``<CURRENT_DIRECTORY/STEM_OF_INPUT>_v2_4.pt``.
