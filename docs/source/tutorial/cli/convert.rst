.. _convert:

Conversion
----------

To convert a trained model from Chemprop v1 to v2, run ``chemprop convert`` and specify:

 * :code:`--input-path <path>` Path of the Chemprop v1 file to convert.
 * :code:`--output-path <path>` Path where the converted Chemprop v2 model will be saved. If unspecified, this will default to ``<CURRENT_DIRECTORY/STEM_OF_INPUT>_v2.pt``.

To convert a trained model from Chemprop v2.0.x to v2.1.y (or newer), run ``chemprop convert --conversion v2_0_to_v2_1`` and additionally specify:

 * :code:`--input-path <path>` Path of the Chemprop v2.0 file to convert.
 * :code:`--output-path <path>` Path where the converted Chemprop v2.1 model will be saved. If unspecified, this will default to ``<CURRENT_DIRECTORY/STEM_OF_INPUT>_v2_1.pt``.

