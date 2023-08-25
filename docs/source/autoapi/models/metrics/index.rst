:py:mod:`models.metrics`
========================

.. py:module:: models.metrics


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.metrics.Metric
   models.metrics.ThresholdedMixin
   models.metrics.MAEMetric
   models.metrics.MSEMetric
   models.metrics.RMSEMetric
   models.metrics.BoundedMixin
   models.metrics.BoundedMAEMetric
   models.metrics.BoundedMSEMetric
   models.metrics.BoundedRMSEMetric
   models.metrics.R2Metric
   models.metrics.AUROCMetric
   models.metrics.AUPRCMetric
   models.metrics.AccuracyMetric
   models.metrics.F1Metric
   models.metrics.BCEMetric
   models.metrics.CrossEntropyMetric
   models.metrics.MCCMetric
   models.metrics.SIDMetric
   models.metrics.WassersteinMetric




Attributes
~~~~~~~~~~

.. autoapisummary::

   models.metrics.MetricRegistry


.. py:data:: MetricRegistry

   

.. py:class:: Metric


   Bases: :py:obj:`chemprop.v2.models.loss.LossFunction`

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


   Bases: :py:obj:`chemprop.v2.models.loss.MSELoss`, :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: RMSEMetric


   Bases: :py:obj:`MSEMetric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(*args, **kwargs)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



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


   Bases: :py:obj:`chemprop.v2.models.loss.BCELoss`, :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: CrossEntropyMetric


   Bases: :py:obj:`chemprop.v2.models.loss.CrossEntropyLoss`, :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: MCCMetric(n_classes, threshold = 0.5, *args)


   Bases: :py:obj:`Metric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:attribute:: minimize
      :value: False

      don't think this works rn

      :type: NOTE(degraff)

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



.. py:class:: SIDMetric


   Bases: :py:obj:`Metric`, :py:obj:`ThresholdedMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, mask, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: WassersteinMetric


   Bases: :py:obj:`Metric`, :py:obj:`ThresholdedMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, mask, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



