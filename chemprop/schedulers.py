import numpy as np
from numpy.typing import ArrayLike
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class NoamLR(LRScheduler):
    r"""A Noam learning rate scheduler schedules the learning rate with a piecewise linear followed
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
        \delta &\mathrel{:=}
            \frac{\mathtt{max\_lr} - \mathtt{init\_lr}}{\mathtt{warmup\_steps}} \\
        \gamma(i) &\mathrel{:=}
            \frac{i - \mathtt{warmup\_steps}}{\mathtt{total\_steps} - \mathtt{warmup\_steps}}


    Parameters
    -----------
    optimizer : Optimizer
        A PyTorch optimizer.
    warmup_epochs : ArrayLike
        The number of epochs during which to linearly increase the learning rate.
    total_epochs : int
        The total number of epochs.
    steps_per_epoch : int
        The number of steps (batches) per epoch.
    init_lr : ArrayLike
        The initial learning rate.
    max_lr : ArrayLike
        The maximum learning rate (achieved after ``warmup_epochs``).
    final_lr : ArrayLike
        The final learning rate (achieved after ``total_epochs``).

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I. "Attention is all you need." Advances in neural information processing systems, 2017, 30. https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: ArrayLike,
        total_epochs: int,
        steps_per_epoch: int,
        init_lrs: ArrayLike,
        max_lrs: ArrayLike,
        final_lrs: ArrayLike,
    ):
        self.num_lrs = len(optimizer.param_groups)
        warmup_epochs = np.atleast_1d(warmup_epochs)
        init_lrs = np.atleast_1d(init_lrs)
        max_lrs = np.atleast_1d(max_lrs)
        self.final_lrs = np.atleast_1d(final_lrs)

        if not (
            self.num_lrs
            == len(warmup_epochs)
            == len(init_lrs)
            == len(max_lrs)
            == len(self.final_lrs)
        ):
            raise ValueError(
                "Number of param groups must match number of: "
                "'warmup_epochs', 'init_lr', 'max_lr', 'final_lr'! "
                f"got: {len(self.optimizer.param_groups)} param groups, "
                f"{len(init_lrs)} init_lr, "
                f"{len(max_lrs)} max_lr, "
                f"{len(self.final_lrs)} final_lr"
            )

        self.current_step = 0
        self.lrs = init_lrs

        warmup_steps = (warmup_epochs * steps_per_epoch).astype(int)
        total_steps = total_epochs * steps_per_epoch
        cooldown_steps = total_steps - warmup_steps

        deltas = (max_lrs - init_lrs) / warmup_steps
        gammas = (self.final_lrs / max_lrs) ** (1 / cooldown_steps)

        self.scheds = []
        for i in range(self.num_lrs):
            warmup = init_lrs[i] + np.arange(warmup_steps[i]) * deltas[i]
            cooldown = max_lrs[i] * (gammas[i] ** np.arange(cooldown_steps[i]))
            self.scheds.append(np.concatenate((warmup, cooldown)))
        self.scheds = np.array(self.scheds)

        super(NoamLR, self).__init__(optimizer)

    def __len__(self) -> int:
        """the number of steps in the learning rate schedule"""
        return self.scheds.shape[1]

    def get_lr(self) -> np.ndarray:
        """Get a list of the current learning rates"""
        return self.lrs

    def step(self, step: int | None = None):
        """Step the learning rate

        Parameters
        ----------
        step : int | None, default=None
            What step to set the learning rate to. If ``None``, use ``self.current_step + 1``.
        """
        self.current_step = step if step is not None else self.current_step + 1

        for i in range(self.num_lrs):
            if self.current_step < len(self):
                self.lrs[i] = self.scheds[i][self.current_step]
            else:
                self.lrs[i] = self.final_lrs[i]

            self.optimizer.param_groups[i]["lr"] = self.lrs[i]
