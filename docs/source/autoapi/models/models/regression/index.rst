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

   .. py:method:: predict_step(*args, **kwargs) -> tuple[torch.Tensor, Ellipsis]



.. py:class:: MveRegressionMPNN(*args, **kwargs)


   Bases: :py:obj:`RegressionMPNN`

   .. py:property:: n_targets
      :type: int


   .. py:method:: forward(inputs, X_f) -> torch.Tensor


   .. py:method:: predict_step(*args, **kwargs) -> tuple[torch.Tensor, Ellipsis]



.. py:class:: EvidentialMPNN(*args, **kwargs)


   Bases: :py:obj:`RegressionMPNN`

   .. py:property:: n_targets
      :type: int


   .. py:method:: forward(inputs, X_f) -> torch.Tensor


   .. py:method:: predict_step(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]



