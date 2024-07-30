from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class NoamLR(LambdaLR):
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
    warmup_epochs : int
        The number of epochs during which to linearly increase the learning rate.
    total_epochs : int
        The total number of epochs.
    steps_per_epoch : int
        The number of steps (batches) per epoch.
    init_lr : float
        The initial learning rate.
    max_lr : float
        The maximum learning rate (achieved after ``warmup_epochs``).
    final_lr : float
        The final learning rate (achieved after ``total_epochs``).

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I. "Attention is all you need." Advances in neural information processing systems, 2017, 30. https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        init_lr: float,
        max_lr: float,
        final_lr: float,
    ):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch

        super().__init__(
            optimizer,
            partial(
                self.lr_lambda,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                init_lr=init_lr,
                max_lr=max_lr,
                final_lr=final_lr,
            ),
        )

    @staticmethod
    def lr_lambda(step, warmup_steps, total_steps, init_lr, max_lr, final_lr):
        if step < warmup_steps:
            delta = (max_lr - init_lr) / warmup_steps
            return step * delta / init_lr + 1
        else:
            gammas = (final_lr / max_lr) ** (1 / (total_steps - warmup_steps))
            return max_lr / init_lr * (gammas ** (step - warmup_steps))
