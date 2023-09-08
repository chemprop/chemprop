:py:mod:`models.modules.readout`
================================

.. py:module:: models.modules.readout


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.modules.readout.Readout
   models.modules.readout.ReadoutFFNBase
   models.modules.readout.RegressionFFN
   models.modules.readout.MveFFN
   models.modules.readout.EvidentialFFN
   models.modules.readout.BinaryClassificationFFNBase
   models.modules.readout.BinaryClassificationFFN
   models.modules.readout.BinaryDirichletFFN
   models.modules.readout.MulticlassClassificationFFN
   models.modules.readout.MulticlassDirichletFFN
   models.modules.readout.SpectralFFN




Attributes
~~~~~~~~~~

.. autoapisummary::

   models.modules.readout.ReadoutRegistry


.. py:data:: ReadoutRegistry

   

.. py:class:: Readout(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`_ReadoutProto`, :py:obj:`chemprop.v2.models.hparams.HasHParams`

   A :class:`Readout` is a protocol that defines a fully differentiable function which maps a tensor of shape `N x d_i` to a tensor of shape `N x d_o`


.. py:class:: ReadoutFFNBase(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`Readout`, :py:obj:`lightning.pytorch.core.mixins.HyperparametersMixin`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:property:: input_dim
      :type: int


   .. py:property:: output_dim
      :type: int


   .. py:property:: n_tasks
      :type: int


   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: RegressionFFN(*args, loc = 0, scale = 1, **kwargs)


   Bases: :py:obj:`ReadoutFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: MveFFN(*args, loc = 0, scale = 1, **kwargs)


   Bases: :py:obj:`RegressionFFN`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 2

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: EvidentialFFN(*args, loc = 0, scale = 1, **kwargs)


   Bases: :py:obj:`RegressionFFN`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 4

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: BinaryClassificationFFNBase(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`ReadoutFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.


.. py:class:: BinaryClassificationFFN(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`BinaryClassificationFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: BinaryDirichletFFN(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`BinaryClassificationFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 2

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: MulticlassClassificationFFN(n_classes, n_tasks = 1, *args, **kwargs)


   Bases: :py:obj:`ReadoutFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: MulticlassDirichletFFN(n_classes, n_tasks = 1, *args, **kwargs)


   Bases: :py:obj:`MulticlassClassificationFFN`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: SpectralFFN(*args, spectral_activation = 'softplus', **kwargs)


   Bases: :py:obj:`ReadoutFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      


