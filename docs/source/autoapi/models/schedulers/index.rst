:py:mod:`models.schedulers`
===========================

.. py:module:: models.schedulers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.schedulers.NoamLR




.. py:class:: NoamLR(optimizer, warmup_epochs, total_epochs, steps_per_epoch, init_lrs, max_lrs, final_lrs)


   Bases: :py:obj:`torch.optim.lr_scheduler.LRScheduler`

   A Noam learning rate scheduler schedules the learning rate with a piecewise linear followed
   by an exponential decay.

   The learning rate increases linearly from ``init_lr`` to ``max_lr`` over the course of
   the first warmup_steps then decreases exponentially to ``final_lr`` over the course of the
   remaining ``total_steps - warmup_steps`` (where ``total_steps = total_epochs * steps_per_epoch``). This is roughly based on the learning rate schedule from [1]_, section 5.3.

   Formally, the learning rate schedule is defined as:

   .. math::
       \mathtt{lr}(i) &=
           \begin{cases}
               \mathtt{init\_lr} + \delta \cdot i &\text{if } i < \mathtt{warmup\_steps} \\
               \mathtt{max\_lr} \cdot \left( \frac{\mathtt{final\_lr}}{\mathtt{max\_lr}} \right)^{\gamma(i)} &\text{otherwise} \\
           \end{cases}
       \\
       \delta &\mathrel{\unicode{x2254}}
           \frac{\mathtt{max\_lr} - \mathtt{init\_lr}}{\mathtt{warmup\_steps}} \\
       \gamma(i) &\mathrel{\unicode{x2254}}
           \frac{i - \mathtt{warmup\_steps}}{\mathtt{total\_steps} - \mathtt{warmup\_steps}}


   :param optimizer: A PyTorch optimizer.
   :type optimizer: Optimizer
   :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
   :type warmup_epochs: ArrayLike
   :param total_epochs: The total number of epochs.
   :type total_epochs: int
   :param steps_per_epoch: The number of steps (batches) per epoch.
   :type steps_per_epoch: int
   :param init_lr: The initial learning rate.
   :type init_lr: ArrayLike
   :param max_lr: The maximum learning rate (achieved after ``warmup_epochs``).
   :type max_lr: ArrayLike
   :param final_lr: The final learning rate (achieved after ``total_epochs``).
   :type final_lr: ArrayLike

   .. rubric:: References

   .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I. "Attention is all you need." Advances in neural information processing systems, 2017, 30. https://arxiv.org/abs/1706.03762

   .. py:method:: __len__()

      the number of steps in the learning rate schedule


   .. py:method:: get_lr()

      Get a list of the current learning rates


   .. py:method:: step(step = None)

      Step the learning rate

      :param step: What step to set the learning rate to. If ``None``, use ``self.current_step + 1``.
      :type step: int | None, default=None



