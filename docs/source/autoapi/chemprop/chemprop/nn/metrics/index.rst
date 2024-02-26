:py:mod:`chemprop.chemprop.nn.metrics`
======================================

.. py:module:: chemprop.chemprop.nn.metrics


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.metrics.Metric
   chemprop.chemprop.nn.metrics.ThresholdedMixin
   chemprop.chemprop.nn.metrics.MAEMetric
   chemprop.chemprop.nn.metrics.MSEMetric
   chemprop.chemprop.nn.metrics.RMSEMetric
   chemprop.chemprop.nn.metrics.BoundedMixin
   chemprop.chemprop.nn.metrics.BoundedMAEMetric
   chemprop.chemprop.nn.metrics.BoundedMSEMetric
   chemprop.chemprop.nn.metrics.BoundedRMSEMetric
   chemprop.chemprop.nn.metrics.R2Metric
   chemprop.chemprop.nn.metrics.AUROCMetric
   chemprop.chemprop.nn.metrics.AUPRCMetric
   chemprop.chemprop.nn.metrics.AccuracyMetric
   chemprop.chemprop.nn.metrics.F1Metric
   chemprop.chemprop.nn.metrics.BCEMetric
   chemprop.chemprop.nn.metrics.CrossEntropyMetric
   chemprop.chemprop.nn.metrics.BinaryMCCMetric
   chemprop.chemprop.nn.metrics.MulticlassMCCMetric
   chemprop.chemprop.nn.metrics.SIDMetric
   chemprop.chemprop.nn.metrics.WassersteinMetric




Attributes
~~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.metrics.MetricRegistry


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


