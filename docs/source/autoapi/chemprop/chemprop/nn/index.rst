:py:mod:`chemprop.chemprop.nn`
==============================

.. py:module:: chemprop.chemprop.nn


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   message_passing/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   agg/index.rst
   ffn/index.rst
   hparams/index.rst
   loss/index.rst
   metrics/index.rst
   predictors/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.Aggregation
   chemprop.chemprop.nn.MeanAggregation
   chemprop.chemprop.nn.SumAggregation
   chemprop.chemprop.nn.NormAggregation
   chemprop.chemprop.nn.AttentiveAggregation
   chemprop.chemprop.nn.LossFunction
   chemprop.chemprop.nn.MSELoss
   chemprop.chemprop.nn.BoundedMSELoss
   chemprop.chemprop.nn.MVELoss
   chemprop.chemprop.nn.EvidentialLoss
   chemprop.chemprop.nn.BCELoss
   chemprop.chemprop.nn.CrossEntropyLoss
   chemprop.chemprop.nn.MccMixin
   chemprop.chemprop.nn.BinaryMCCLoss
   chemprop.chemprop.nn.MulticlassMCCLoss
   chemprop.chemprop.nn.DirichletMixin
   chemprop.chemprop.nn.BinaryDirichletLoss
   chemprop.chemprop.nn.MulticlassDirichletLoss
   chemprop.chemprop.nn.SIDLoss
   chemprop.chemprop.nn.WassersteinLoss
   chemprop.chemprop.nn.Metric
   chemprop.chemprop.nn.ThresholdedMixin
   chemprop.chemprop.nn.MAEMetric
   chemprop.chemprop.nn.MSEMetric
   chemprop.chemprop.nn.RMSEMetric
   chemprop.chemprop.nn.BoundedMixin
   chemprop.chemprop.nn.BoundedMAEMetric
   chemprop.chemprop.nn.BoundedMSEMetric
   chemprop.chemprop.nn.BoundedRMSEMetric
   chemprop.chemprop.nn.R2Metric
   chemprop.chemprop.nn.AUROCMetric
   chemprop.chemprop.nn.AUPRCMetric
   chemprop.chemprop.nn.AccuracyMetric
   chemprop.chemprop.nn.F1Metric
   chemprop.chemprop.nn.BCEMetric
   chemprop.chemprop.nn.CrossEntropyMetric
   chemprop.chemprop.nn.BinaryMCCMetric
   chemprop.chemprop.nn.MulticlassMCCMetric
   chemprop.chemprop.nn.SIDMetric
   chemprop.chemprop.nn.WassersteinMetric
   chemprop.chemprop.nn.MessagePassing
   chemprop.chemprop.nn.AtomMessagePassing
   chemprop.chemprop.nn.BondMessagePassing
   chemprop.chemprop.nn.MulticomponentMessagePassing
   chemprop.chemprop.nn.Predictor
   chemprop.chemprop.nn.RegressionFFN
   chemprop.chemprop.nn.MveFFN
   chemprop.chemprop.nn.EvidentialFFN
   chemprop.chemprop.nn.BinaryClassificationFFNBase
   chemprop.chemprop.nn.BinaryClassificationFFN
   chemprop.chemprop.nn.BinaryDirichletFFN
   chemprop.chemprop.nn.MulticlassClassificationFFN
   chemprop.chemprop.nn.MulticlassDirichletFFN
   chemprop.chemprop.nn.SpectralFFN
   chemprop.chemprop.nn.Activation




Attributes
~~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.AggregationRegistry
   chemprop.chemprop.nn.LossFunctionRegistry
   chemprop.chemprop.nn.MetricRegistry
   chemprop.chemprop.nn.PredictorRegistry


.. py:class:: Aggregation(dim = 0, *args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`chemprop.nn.hparams.HasHParams`

   An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
   a batch of graph-level representations

   .. note::
       this class is abstract and cannot be instantiated.

   .. seealso:: :class:`~chemprop.v2.models.modules.agg.MeanAggregation`, :class:`~chemprop.v2.models.modules.agg.SumAggregation`, :class:`~chemprop.v2.models.modules.agg.NormAggregation`

   .. py:method:: forward(H, batch)
      :abstractmethod:

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``V x d`` containing the batched node-level representations of ``n``
                graphs
      :type H: Tensor
      :param batch: a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to
      :type batch: Tensor

      :returns: a tensor of shape ``n x d`` containing the graph-level representations
      :rtype: Tensor



.. py:data:: AggregationRegistry

   

.. py:class:: MeanAggregation(dim = 0, *args, **kwargs)


   Bases: :py:obj:`Aggregation`

   Average the graph-level representation:

   .. math::
       \mathbf h = \frac{1}{|V|} \sum_{v \in V} \mathbf h_v

   .. py:method:: forward(H, batch)

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``V x d`` containing the batched node-level representations of ``n``
                graphs
      :type H: Tensor
      :param batch: a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to
      :type batch: Tensor

      :returns: a tensor of shape ``n x d`` containing the graph-level representations
      :rtype: Tensor



.. py:class:: SumAggregation(dim = 0, *args, **kwargs)


   Bases: :py:obj:`Aggregation`

   Sum the graph-level representation:

   .. math::
       \mathbf h = \sum_{v \in V} \mathbf h_v


   .. py:method:: forward(H, batch)

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``V x d`` containing the batched node-level representations of ``n``
                graphs
      :type H: Tensor
      :param batch: a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to
      :type batch: Tensor

      :returns: a tensor of shape ``n x d`` containing the graph-level representations
      :rtype: Tensor



.. py:class:: NormAggregation(dim = 0, *args, norm = 100, **kwargs)


   Bases: :py:obj:`SumAggregation`

   Sum the graph-level representation and divide by a normalization constant:

   .. math::
       \mathbf h = \frac{1}{c} \sum_{v \in V} \mathbf h_v

   .. py:method:: forward(H, batch)

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``V x d`` containing the batched node-level representations of ``n``
                graphs
      :type H: Tensor
      :param batch: a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to
      :type batch: Tensor

      :returns: a tensor of shape ``n x d`` containing the graph-level representations
      :rtype: Tensor



.. py:class:: AttentiveAggregation(dim = 0, *args, output_size, **kwargs)


   Bases: :py:obj:`Aggregation`

   An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
   a batch of graph-level representations

   .. note::
       this class is abstract and cannot be instantiated.

   .. seealso:: :class:`~chemprop.v2.models.modules.agg.MeanAggregation`, :class:`~chemprop.v2.models.modules.agg.SumAggregation`, :class:`~chemprop.v2.models.modules.agg.NormAggregation`

   .. py:method:: forward(H, batch)

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``V x d`` containing the batched node-level representations of ``n``
                graphs
      :type H: Tensor
      :param batch: a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to
      :type batch: Tensor

      :returns: a tensor of shape ``n x d`` containing the graph-level representations
      :rtype: Tensor



.. py:class:: LossFunction


   Bases: :py:obj:`abc.ABC`, :py:obj:`chemprop.utils.ReprMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: __call__(preds, targets, mask, w_s, w_t, lt_mask, gt_mask)

      Calculate the mean loss function value given predicted and target values

      :param preds: a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
                    `b x t x c` (multiclass classification) containing the predictions, where `b` is the
                    batch size, `t` is the number of tasks to predict, `s` is the number of
                    targets to predict for each task, and `c` is the number of classes.
      :type preds: Tensor
      :param targets: a float tensor of shape `b x t` containing the target values
      :type targets: Tensor
      :param mask: a boolean tensor of shape `b x t` indicating whether the given prediction should be
                   included in the loss calculation
      :type mask: Tensor
      :param w_s: a tensor of shape `b` or `b x 1` containing the per-sample weight
      :type w_s: Tensor
      :param w_t: a tensor of shape `t` or `1 x t` containing the per-task weight
      :type w_t: Tensor
      :param lt_mask:
      :type lt_mask: Tensor
      :param gt_mask:
      :type gt_mask: Tensor

      :returns: a scalar containing the fully reduced loss
      :rtype: Tensor


   .. py:method:: forward(preds, targets, mask, w_s, w_t, lt_mask, gt_mask)
      :abstractmethod:

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:data:: LossFunctionRegistry

   

.. py:class:: MSELoss


   Bases: :py:obj:`LossFunction`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: BoundedMSELoss


   Bases: :py:obj:`MSELoss`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, mask, w_s, w_t, lt_mask, gt_mask)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: MVELoss


   Bases: :py:obj:`LossFunction`

   Calculate the loss using Eq. 9 from [nix1994]_

   .. rubric:: References

   .. [nix1994] Nix, D. A.; Weigend, A. S. "Estimating the mean and variance of the target
       probability distribution." Proceedings of 1994 IEEE International Conference on Neural
       Networks, 1994 https://doi.org/10.1109/icnn.1994.374138

   .. py:method:: forward(preds, targets, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: EvidentialLoss(v_kl = 0.2, eps = 1e-08)


   Bases: :py:obj:`LossFunction`

   Caculate the loss using Eq. **TODO** from [soleimany2021]_

   .. rubric:: References

   .. [soleimany2021] Soleimany, A.P.; Amini, A.; Goldman, S.; Rus, D.; Bhatia, S.N.; Coley, C.W.;
       "Evidential Deep Learning for Guided Molecular Property Prediction and Discovery." ACS
       Cent. Sci. 2021, 7, 8, 1356-1367. https://doi.org/10.1021/acscentsci.1c00546

   .. py:method:: forward(preds, targets, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.


   .. py:method:: get_params()



.. py:class:: BCELoss


   Bases: :py:obj:`LossFunction`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: CrossEntropyLoss


   Bases: :py:obj:`LossFunction`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: MccMixin


   Calculate a soft Matthews correlation coefficient ([mccWiki]_) loss for multiclass
   classification based on the implementataion of [mccSklearn]_

   .. rubric:: References

   .. [mccWiki] https://en.wikipedia.org/wiki/Phi_coefficient#Multiclass_case
   .. [mccSklearn] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html

   .. py:method:: __call__(preds, targets, mask, w_s, w_t, *args)



.. py:class:: BinaryMCCLoss


   Bases: :py:obj:`LossFunction`, :py:obj:`MccMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, mask, w_s, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: MulticlassMCCLoss


   Bases: :py:obj:`LossFunction`, :py:obj:`MccMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, mask, w_s, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: DirichletMixin(v_kl = 0.2)


   Uses the loss function from [sensoy2018]_ based on the implementation at [sensoyGithub]_

   .. rubric:: References

   .. [sensoy2018] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
       classification uncertainty." NeurIPS, 2018, 31. https://doi.org/10.48550/arXiv.1806.01768
   .. [sensoyGithub] https://muratsensoy.github.io/uncertainty.html#Define-the-loss-function

   .. py:method:: forward(preds, targets, *args)


   .. py:method:: get_params()



.. py:class:: BinaryDirichletLoss(v_kl = 0.2)


   Bases: :py:obj:`DirichletMixin`, :py:obj:`LossFunction`

   Uses the loss function from [sensoy2018]_ based on the implementation at [sensoyGithub]_

   .. rubric:: References

   .. [sensoy2018] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
       classification uncertainty." NeurIPS, 2018, 31. https://doi.org/10.48550/arXiv.1806.01768
   .. [sensoyGithub] https://muratsensoy.github.io/uncertainty.html#Define-the-loss-function

   .. py:method:: forward(preds, targets, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: MulticlassDirichletLoss(v_kl = 0.2)


   Bases: :py:obj:`DirichletMixin`, :py:obj:`LossFunction`

   Uses the loss function from [sensoy2018]_ based on the implementation at [sensoyGithub]_

   .. rubric:: References

   .. [sensoy2018] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
       classification uncertainty." NeurIPS, 2018, 31. https://doi.org/10.48550/arXiv.1806.01768
   .. [sensoyGithub] https://muratsensoy.github.io/uncertainty.html#Define-the-loss-function

   .. py:method:: forward(preds, targets, mask, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: SIDLoss


   Bases: :py:obj:`LossFunction`, :py:obj:`_ThresholdMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, mask, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: WassersteinLoss


   Bases: :py:obj:`LossFunction`, :py:obj:`_ThresholdMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, mask, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: Metric


   Bases: :py:obj:`chemprop.nn.loss.LossFunction`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:attribute:: minimize
      :type: bool
      :value: True

      

   .. py:method:: __call__(preds, targets, mask, w_s, w_t, lt_mask, gt_mask)

      Calculate the mean loss function value given predicted and target values

      :param preds: a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
                    `b x t x c` (multiclass classification) containing the predictions, where `b` is the
                    batch size, `t` is the number of tasks to predict, `s` is the number of
                    targets to predict for each task, and `c` is the number of classes.
      :type preds: Tensor
      :param targets: a float tensor of shape `b x t` containing the target values
      :type targets: Tensor
      :param mask: a boolean tensor of shape `b x t` indicating whether the given prediction should be
                   included in the loss calculation
      :type mask: Tensor
      :param w_s: a tensor of shape `b` or `b x 1` containing the per-sample weight
      :type w_s: Tensor
      :param w_t: a tensor of shape `t` or `1 x t` containing the per-task weight
      :type w_t: Tensor
      :param lt_mask:
      :type lt_mask: Tensor
      :param gt_mask:
      :type gt_mask: Tensor

      :returns: a scalar containing the fully reduced loss
      :rtype: Tensor


   .. py:method:: forward(preds, targets, mask, lt_mask, gt_mask)
      :abstractmethod:

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:data:: MetricRegistry

   

.. py:class:: ThresholdedMixin


   .. py:attribute:: threshold
      :type: float | None
      :value: 0.5

      


.. py:class:: MAEMetric


   Bases: :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: MSEMetric


   Bases: :py:obj:`chemprop.nn.loss.MSELoss`, :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: RMSEMetric


   Bases: :py:obj:`MSEMetric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: __call__(preds, targets, mask, w_s, w_t, lt_mask, gt_mask)

      Calculate the mean loss function value given predicted and target values

      :param preds: a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
                    `b x t x c` (multiclass classification) containing the predictions, where `b` is the
                    batch size, `t` is the number of tasks to predict, `s` is the number of
                    targets to predict for each task, and `c` is the number of classes.
      :type preds: Tensor
      :param targets: a float tensor of shape `b x t` containing the target values
      :type targets: Tensor
      :param mask: a boolean tensor of shape `b x t` indicating whether the given prediction should be
                   included in the loss calculation
      :type mask: Tensor
      :param w_s: a tensor of shape `b` or `b x 1` containing the per-sample weight
      :type w_s: Tensor
      :param w_t: a tensor of shape `t` or `1 x t` containing the per-task weight
      :type w_t: Tensor
      :param lt_mask:
      :type lt_mask: Tensor
      :param gt_mask:
      :type gt_mask: Tensor

      :returns: a scalar containing the fully reduced loss
      :rtype: Tensor



.. py:class:: BoundedMixin


   .. py:method:: forward(preds, targets, mask, lt_mask, gt_mask)



.. py:class:: BoundedMAEMetric


   Bases: :py:obj:`MAEMetric`, :py:obj:`BoundedMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: BoundedMSEMetric


   Bases: :py:obj:`MSEMetric`, :py:obj:`BoundedMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: BoundedRMSEMetric


   Bases: :py:obj:`RMSEMetric`, :py:obj:`BoundedMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: R2Metric


   Bases: :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:attribute:: minimize
      :value: False

      

   .. py:method:: __call__(preds, targets, mask, *args, **kwargs)

      Calculate the mean loss function value given predicted and target values

      :param preds: a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
                    `b x t x c` (multiclass classification) containing the predictions, where `b` is the
                    batch size, `t` is the number of tasks to predict, `s` is the number of
                    targets to predict for each task, and `c` is the number of classes.
      :type preds: Tensor
      :param targets: a float tensor of shape `b x t` containing the target values
      :type targets: Tensor
      :param mask: a boolean tensor of shape `b x t` indicating whether the given prediction should be
                   included in the loss calculation
      :type mask: Tensor
      :param w_s: a tensor of shape `b` or `b x 1` containing the per-sample weight
      :type w_s: Tensor
      :param w_t: a tensor of shape `t` or `1 x t` containing the per-task weight
      :type w_t: Tensor
      :param lt_mask:
      :type lt_mask: Tensor
      :param gt_mask:
      :type gt_mask: Tensor

      :returns: a scalar containing the fully reduced loss
      :rtype: Tensor



.. py:class:: AUROCMetric


   Bases: :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:attribute:: minimize
      :value: False

      

   .. py:method:: __call__(preds, targets, mask, *args, **kwargs)

      Calculate the mean loss function value given predicted and target values

      :param preds: a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
                    `b x t x c` (multiclass classification) containing the predictions, where `b` is the
                    batch size, `t` is the number of tasks to predict, `s` is the number of
                    targets to predict for each task, and `c` is the number of classes.
      :type preds: Tensor
      :param targets: a float tensor of shape `b x t` containing the target values
      :type targets: Tensor
      :param mask: a boolean tensor of shape `b x t` indicating whether the given prediction should be
                   included in the loss calculation
      :type mask: Tensor
      :param w_s: a tensor of shape `b` or `b x 1` containing the per-sample weight
      :type w_s: Tensor
      :param w_t: a tensor of shape `t` or `1 x t` containing the per-task weight
      :type w_t: Tensor
      :param lt_mask:
      :type lt_mask: Tensor
      :param gt_mask:
      :type gt_mask: Tensor

      :returns: a scalar containing the fully reduced loss
      :rtype: Tensor



.. py:class:: AUPRCMetric


   Bases: :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:attribute:: minimize
      :value: False

      

   .. py:method:: __call__(preds, targets, *args, **kwargs)

      Calculate the mean loss function value given predicted and target values

      :param preds: a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
                    `b x t x c` (multiclass classification) containing the predictions, where `b` is the
                    batch size, `t` is the number of tasks to predict, `s` is the number of
                    targets to predict for each task, and `c` is the number of classes.
      :type preds: Tensor
      :param targets: a float tensor of shape `b x t` containing the target values
      :type targets: Tensor
      :param mask: a boolean tensor of shape `b x t` indicating whether the given prediction should be
                   included in the loss calculation
      :type mask: Tensor
      :param w_s: a tensor of shape `b` or `b x 1` containing the per-sample weight
      :type w_s: Tensor
      :param w_t: a tensor of shape `t` or `1 x t` containing the per-task weight
      :type w_t: Tensor
      :param lt_mask:
      :type lt_mask: Tensor
      :param gt_mask:
      :type gt_mask: Tensor

      :returns: a scalar containing the fully reduced loss
      :rtype: Tensor



.. py:class:: AccuracyMetric


   Bases: :py:obj:`Metric`, :py:obj:`ThresholdedMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:attribute:: minimize
      :value: False

      

   .. py:method:: __call__(preds, targets, mask, *args, **kwargs)

      Calculate the mean loss function value given predicted and target values

      :param preds: a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
                    `b x t x c` (multiclass classification) containing the predictions, where `b` is the
                    batch size, `t` is the number of tasks to predict, `s` is the number of
                    targets to predict for each task, and `c` is the number of classes.
      :type preds: Tensor
      :param targets: a float tensor of shape `b x t` containing the target values
      :type targets: Tensor
      :param mask: a boolean tensor of shape `b x t` indicating whether the given prediction should be
                   included in the loss calculation
      :type mask: Tensor
      :param w_s: a tensor of shape `b` or `b x 1` containing the per-sample weight
      :type w_s: Tensor
      :param w_t: a tensor of shape `t` or `1 x t` containing the per-task weight
      :type w_t: Tensor
      :param lt_mask:
      :type lt_mask: Tensor
      :param gt_mask:
      :type gt_mask: Tensor

      :returns: a scalar containing the fully reduced loss
      :rtype: Tensor



.. py:class:: F1Metric


   Bases: :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:attribute:: minimize
      :value: False

      

   .. py:method:: __call__(preds, targets, mask, *args, **kwargs)

      Calculate the mean loss function value given predicted and target values

      :param preds: a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
                    `b x t x c` (multiclass classification) containing the predictions, where `b` is the
                    batch size, `t` is the number of tasks to predict, `s` is the number of
                    targets to predict for each task, and `c` is the number of classes.
      :type preds: Tensor
      :param targets: a float tensor of shape `b x t` containing the target values
      :type targets: Tensor
      :param mask: a boolean tensor of shape `b x t` indicating whether the given prediction should be
                   included in the loss calculation
      :type mask: Tensor
      :param w_s: a tensor of shape `b` or `b x 1` containing the per-sample weight
      :type w_s: Tensor
      :param w_t: a tensor of shape `t` or `1 x t` containing the per-task weight
      :type w_t: Tensor
      :param lt_mask:
      :type lt_mask: Tensor
      :param gt_mask:
      :type gt_mask: Tensor

      :returns: a scalar containing the fully reduced loss
      :rtype: Tensor



.. py:class:: BCEMetric


   Bases: :py:obj:`chemprop.nn.loss.BCELoss`, :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: CrossEntropyMetric


   Bases: :py:obj:`chemprop.nn.loss.CrossEntropyLoss`, :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: BinaryMCCMetric


   Bases: :py:obj:`chemprop.nn.loss.BinaryMCCLoss`, :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: MulticlassMCCMetric


   Bases: :py:obj:`chemprop.nn.loss.MulticlassMCCLoss`, :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: SIDMetric


   Bases: :py:obj:`chemprop.nn.loss.SIDLoss`, :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: WassersteinMetric


   Bases: :py:obj:`chemprop.nn.loss.WassersteinLoss`, :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: MessagePassing(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`chemprop.nn.hparams.HasHParams`

   A :class:`MessagePassing` module encodes a batch of molecular graphs
   using message passing to learn vertex-level hidden representations.

   .. py:attribute:: input_dim
      :type: int

      

   .. py:attribute:: output_dim
      :type: int

      

   .. py:method:: forward(bmg, V_d = None)
      :abstractmethod:

      Encode a batch of molecular graphs.

      :param bmg: the batch of :class:`~chemprop.featurizers.molgraph.MolGraph`\s to encode
      :type bmg: BatchMolGraph
      :param V_d: an optional tensor of shape `V x d_vd` containing additional descriptors for each atom
                  in the batch. These will be concatenated to the learned atomic descriptors and
                  transformed before the readout phase. NOTE: recall that `V` is equal to `num_atoms + 1`\,
                  so if provided, this tensor must be 0-padded in the 0th row.
      :type V_d: Tensor | None, default=None

      :returns: a tensor of shape `V x d_h` or `V x (d_h + d_vd)` containing the hidden representation
                of each vertex in the batch of graphs. The feature dimension depends on whether
                additional atom descriptors were provided
      :rtype: Tensor



.. py:class:: AtomMessagePassing(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, bias = False, depth = 3, dropout = 0, activation = Activation.RELU, undirected = False, d_vd = None)


   Bases: :py:obj:`_MessagePassingBase`

   A :class:`AtomMessagePassing` encodes a batch of molecular graphs by passing messages along
   atoms.

   It implements the following operation:

   .. math::

       h_v^{(0)} &= \tau \left( \mathbf{W}_i(x_v) \right) \\
       m_v^{(t)} &= \sum_{u \in \mathcal{N}(v)} h_u^{(t-1)} \mathbin\Vert e_{uv} \\
       h_v^{(t)} &= \tau\left(h_v^{(0)} + \mathbf{W}_h m_v^{(t-1)}\right) \\
       m_v^{(T)} &= \sum_{w \in \mathcal{N}(v)} h_w^{(T-1)} \\
       h_v^{(T)} &= \tau \left (\mathbf{W}_o \left( x_v \mathbin\Vert m_{v}^{(T)} \right)  \right),

   where :math:`\tau` is the activation function; :math:`\mathbf{W}_i`, :math:`\mathbf{W}_h`, and
   :math:`\mathbf{W}_o` are learned weight matrices; :math:`e_{vw}` is the feature vector of the
   bond between atoms :math:`v` and :math:`w`; :math:`x_v` is the feature vector of atom :math:`v`;
   :math:`h_v^{(t)}` is the hidden representation of atom :math:`v` at iteration :math:`t`;
   :math:`m_v^{(t)}` is the message received by atom :math:`v` at iteration :math:`t`; and
   :math:`t \in \{1, \dots, T\}` is the number of message passing iterations.

   .. py:method:: setup(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, d_vd = None, bias = False)

      setup the weight matrices used in the message passing update functions

      :param d_v: the vertex feature dimension
      :type d_v: int
      :param d_e: the edge feature dimension
      :type d_e: int
      :param d_h: the hidden dimension during message passing
      :type d_h: int, default=300
      :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden
                   features before readout, if any
      :type d_vd: int | None, default=None
      :param bias: whether to add a learned bias to the matrices
      :type bias: bool, default=False

      :returns: **W_i, W_h, W_o, W_d** -- the input, hidden, output, and descriptor weight matrices, respectively, used in the
                message passing update functions. The descriptor weight matrix is `None` if no vertex
                dimension is supplied
      :rtype: tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]


   .. py:method:: initialize(bmg)

      initialize the message passing scheme by calculating initial matrix of hidden features


   .. py:method:: message(H, bmg)

      Calculate the message matrix



.. py:class:: BondMessagePassing(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, bias = False, depth = 3, dropout = 0, activation = Activation.RELU, undirected = False, d_vd = None)


   Bases: :py:obj:`_MessagePassingBase`

   A :class:`BondMessagePassing` encodes a batch of molecular graphs by passing messages along
   directed bonds.

   It implements the following operation:

   .. math::

       h_{vw}^{(0)} &= \tau \left( \mathbf W_i(e_{vw}) \right) \\
       m_{vw}^{(t)} &= \sum_{u \in \mathcal N(v)\setminus w} h_{uv}^{(t-1)} \\
       h_{vw}^{(t)} &= \tau \left(h_v^{(0)} + \mathbf W_h m_{vw}^{(t-1)} \right) \\
       m_v^{(T)} &= \sum_{w \in \mathcal N(v)} h_w^{(T-1)} \\
       h_v^{(T)} &= \tau \left (\mathbf W_o \left( x_v \mathbin\Vert m_{v}^{(T)} \right) \right),

   where :math:`\tau` is the activation function; :math:`\mathbf W_i`, :math:`\mathbf W_h`, and
   :math:`\mathbf W_o` are learned weight matrices; :math:`e_{vw}` is the feature vector of the
   bond between atoms :math:`v` and :math:`w`; :math:`x_v` is the feature vector of atom :math:`v`;
   :math:`h_{vw}^{(t)}` is the hidden representation of the bond :math:`v \rightarrow w` at
   iteration :math:`t`; :math:`m_{vw}^{(t)}` is the message received by the bond :math:`v
   \to w` at iteration :math:`t`; and :math:`t \in \{1, \dots, T-1\}` is the number of
   message passing iterations.

   .. py:method:: setup(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, d_vd = None, bias = False)

      setup the weight matrices used in the message passing update functions

      :param d_v: the vertex feature dimension
      :type d_v: int
      :param d_e: the edge feature dimension
      :type d_e: int
      :param d_h: the hidden dimension during message passing
      :type d_h: int, default=300
      :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden
                   features before readout, if any
      :type d_vd: int | None, default=None
      :param bias: whether to add a learned bias to the matrices
      :type bias: bool, default=False

      :returns: **W_i, W_h, W_o, W_d** -- the input, hidden, output, and descriptor weight matrices, respectively, used in the
                message passing update functions. The descriptor weight matrix is `None` if no vertex
                dimension is supplied
      :rtype: tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]


   .. py:method:: initialize(bmg)

      initialize the message passing scheme by calculating initial matrix of hidden features


   .. py:method:: message(H, bmg)

      Calculate the message matrix



.. py:class:: MulticomponentMessagePassing(blocks, n_components, shared = False)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`chemprop.nn.hparams.HasHParams`

   A `MulticomponentMessagePassing` performs message-passing on each individual input in a
   multicomponent input then concatenates the representation of each input to construct a
   global representation

   :param blocks: the invidual message-passing blocks for each input
   :type blocks: Sequence[MolecularMessagePassingBlock]
   :param n_components: the number of components in each input
   :type n_components: int
   :param shared: whether one block will be shared among all components in an input. If not, a separate
                  block will be learned for each component.
   :type shared: bool, default=False

   .. py:property:: output_dim
      :type: int


   .. py:method:: __len__()


   .. py:method:: forward(bmgs, V_ds)

      Encode the multicomponent inputs

      :param bmgs:
      :type bmgs: Iterable[BatchMolGraph]
      :param V_ds:
      :type V_ds: Iterable[Tensor | None]

      :returns: a list of tensors of shape `b x d_i` containing the respective encodings of the `i` h component, where `b` is the number of components in the batch, and `d_i` is the output dimension of the `i`   h encoder
      :rtype: list[Tensor]



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

      


.. py:class:: Activation


   Bases: :py:obj:`chemprop.utils.utils.EnumMapping`

   Enum where members are also (and must be) strings

   .. py:attribute:: RELU

      

   .. py:attribute:: LEAKYRELU

      

   .. py:attribute:: PRELU

      

   .. py:attribute:: TANH

      

   .. py:attribute:: SELU

      

   .. py:attribute:: ELU

      


