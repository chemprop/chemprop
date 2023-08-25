:py:mod:`models.models.classification`
======================================

.. py:module:: models.models.classification


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.models.classification.ClassificationMPNN
   models.models.classification.BinaryClassificationMPNN
   models.models.classification.DirichletClassificationMPNN




.. py:class:: ClassificationMPNN


   Bases: :py:obj:`chemprop.v2.models.models.base.MPNN`


.. py:class:: BinaryClassificationMPNN


   Bases: :py:obj:`ClassificationMPNN`

   .. py:method:: predict_step(*args, **kwargs)



.. py:class:: DirichletClassificationMPNN(*args, **kwargs)


   Bases: :py:obj:`ClassificationMPNN`

   .. py:property:: n_targets
      :type: int


   .. py:method:: forward(*args, **kwargs)


   .. py:method:: predict_step(*args, **kwargs)



