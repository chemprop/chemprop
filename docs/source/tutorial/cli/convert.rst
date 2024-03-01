.. _convert:

Conversion
----------

To convert a trained model from Chemprop v1 to v2, run ``chemprop convert`` and specify:

 * :code:`--input-path <path>` Path of the Chemprop v1 file to convert.
 * :code:`--output-path <path>` Path where the converted Chemprop v2 will be saved. If unspecified, this will default to ``<PATH/TO/V1_MODEL>_v2.ckpt``.

