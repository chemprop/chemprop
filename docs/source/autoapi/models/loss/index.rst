:py:mod:`models.loss`
=====================

.. py:module:: models.loss


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.loss.LossFunction
   models.loss.MSELoss
   models.loss.BoundedMSELoss
   models.loss.MVELoss
   models.loss.EvidentialLoss
   models.loss.BCELoss
   models.loss.CrossEntropyLoss
   models.loss.MccMixin
   models.loss.BinaryMCCLoss
   models.loss.MulticlassMCCLoss
   models.loss.DirichletMixin
   models.loss.BinaryDirichletLoss
   models.loss.MulticlassDirichletLoss
   models.loss.SIDLoss
   models.loss.WassersteinLoss




Attributes
~~~~~~~~~~

.. autoapisummary::

   models.loss.LossFunctionRegistry


.. py:data:: LossFunctionRegistry

   

.. py:class:: LossFunction


   Bases: :py:obj:`abc.ABC`, :py:obj:`chemprop.v2.utils.ReprMixin`

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

   .. py:method:: forward(preds, targets, *args, lt_mask, gt_mask)

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


   Bases: :py:obj:`LossFunction`, :py:obj:`DirichletMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: forward(preds, targets, *args)

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: MulticlassDirichletLoss(v_kl = 0.2)


   Bases: :py:obj:`LossFunction`, :py:obj:`DirichletMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.

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



