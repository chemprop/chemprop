:py:mod:`chemprop.chemprop.nn.predictors`
=========================================

.. py:module:: chemprop.chemprop.nn.predictors


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.predictors.Predictor
   chemprop.chemprop.nn.predictors.RegressionFFN
   chemprop.chemprop.nn.predictors.MveFFN
   chemprop.chemprop.nn.predictors.EvidentialFFN
   chemprop.chemprop.nn.predictors.BinaryClassificationFFNBase
   chemprop.chemprop.nn.predictors.BinaryClassificationFFN
   chemprop.chemprop.nn.predictors.BinaryDirichletFFN
   chemprop.chemprop.nn.predictors.MulticlassClassificationFFN
   chemprop.chemprop.nn.predictors.MulticlassDirichletFFN
   chemprop.chemprop.nn.predictors.SpectralFFN




Attributes
~~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.predictors.PredictorRegistry


.. py:class:: Predictor(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`chemprop.nn.hparams.HasHParams`

   A :class:`Predictor` is a protocol that defines a differentiable function
   :math:`f : \mathbb R^d \mapsto \mathbb R^o

   .. py:attribute:: input_dim
      :type: int

      the input dimension

   .. py:attribute:: output_dim
      :type: int

      the output dimension

   .. py:attribute:: n_tasks
      :type: int

      the number of tasks `t` to predict for each input

   .. py:attribute:: n_targets
      :type: int

      the number of targets `s` to predict for each task `t`

   .. py:attribute:: criterion
      :type: chemprop.nn.metrics.LossFunction

      the loss function to use for training

   .. py:method:: forward(Z)
      :abstractmethod:


   .. py:method:: train_step(Z)
      :abstractmethod:



.. py:data:: PredictorRegistry

   

.. py:class:: RegressionFFN(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None, loc = 0, scale = 1)


   Bases: :py:obj:`_FFNPredictorBase`

   A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
   underlying :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: MveFFN(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None, loc = 0, scale = 1)


   Bases: :py:obj:`RegressionFFN`

   A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
   underlying :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 2

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: EvidentialFFN(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None, loc = 0, scale = 1)


   Bases: :py:obj:`RegressionFFN`

   A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
   underlying :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 4

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: BinaryClassificationFFNBase(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`_FFNPredictorBase`

   A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
   underlying :class:`SimpleFFN` to map the learned fingerprint to the desired output.


.. py:class:: BinaryClassificationFFN(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`BinaryClassificationFFNBase`

   A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
   underlying :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: BinaryDirichletFFN(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`BinaryClassificationFFNBase`

   A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
   underlying :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 2

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: MulticlassClassificationFFN(n_classes, n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`_FFNPredictorBase`

   A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
   underlying :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: MulticlassDirichletFFN(n_classes, n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`MulticlassClassificationFFN`

   A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
   underlying :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: SpectralFFN(*args, spectral_activation = 'softplus', **kwargs)


   Bases: :py:obj:`_FFNPredictorBase`

   A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
   underlying :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      


