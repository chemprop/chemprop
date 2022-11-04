from __future__ import annotations

from typing import Union

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """A Noam learning rate scheduler schedules the learning rate with a piecewise linear followed
    by an exponential decay.

    The learning rate increases linearly from `init_lr` to `max_lr` over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from [1]_, section 5.3.

    Parameters
    -----------
    optimizer : Optimizer
        A PyTorch optimizer.
    warmup_epochs : list[Union[float, int]]
        The number of epochs during which to linearly increase the learning rate.
    total_epochs : list[int]
        The total number of epochs.
    steps_per_epoch : int
        The number of steps (batches) per epoch.
    init_lr : list[float]
        The initial learning rate.
    max_lr : list[float]
        The maximum learning rate (achieved after :code:`warmup_epochs`).
    final_lr : list[float]
        The final learning rate (achieved after :code:`total_epochs`).

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I. "Attention is all you need." Advances in neural information processing systems, 2017, 30. https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: list[Union[float, int]],
        total_epochs: list[int],
        steps_per_epoch: int,
        init_lr: list[float],
        max_lr: list[float],
        final_lr: list[float],
    ):
        lengths = np.array(
            [
                len(optimizer.param_groups),
                len(warmup_epochs),
                len(total_epochs),
                len(init_lr),
                len(max_lr),
                len(final_lr),
            ]
        )
        if not (np.diff(lengths) == 0).all():
            raise ValueError(
                "Number of param groups must match length of: "
                "warmup_epochs, total_epochs, steps_per_epoch, init_lr, max_lr, final_lr! "
                f"got: {lengths[0]} param groups and respective lengths: {lengths[1:].tolist()}, "
            )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        cooldown_steps = self.total_steps - self.warmup_steps
        self.gamma = (self.final_lr / self.max_lr) ** (1 / cooldown_steps)

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Get a list of the current learning rates"""
        return list(self.lr)

    def step(self, current_step: int = None):
        """Step the learning rate

        Parameters
        ----------
        current_step : Optional[int], default=None
            What step to set the learning rate to. If None, :code:`current_step += 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                decay_term = self.gamma[i] ** (self.current_step - self.warmup_steps[i])
                self.lr[i] = self.max_lr[i] * decay_term
            else:
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]["lr"] = self.lr[i]
