:py:mod:`models.models.regression`
==================================

.. py:module:: models.models.regression


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.models.regression.RegressionMPNN
   models.models.regression.MveRegressionMPNN
   models.models.regression.EvidentialMPNN




.. py:class:: RegressionMPNN(*args, **kwargs)


   Bases: :py:obj:`chemprop.v2.models.models.base.MPNN`

   .. py:method:: predict_step(*args, **kwargs)



.. py:class:: MveRegressionMPNN(*args, **kwargs)


   Bases: :py:obj:`RegressionMPNN`

   .. py:property:: n_targets
      :type: int


   .. py:method:: forward(inputs, X_f)


   .. py:method:: predict_step(*args, **kwargs)



.. py:class:: EvidentialMPNN(*args, **kwargs)


   Bases: :py:obj:`RegressionMPNN`

   .. py:property:: n_targets
      :type: int


   .. py:method:: forward(inputs, X_f)


   .. py:method:: predict_step(*args, **kwargs)



