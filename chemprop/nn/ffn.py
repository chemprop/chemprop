from abc import abstractmethod

from torch import Tensor, nn

from chemprop.nn.utils import get_activation_function


class FFN(nn.Module):
    r"""A :class:`FFN` is a differentiable function
    :math:`f_\theta : \mathbb R^i \mapsto \mathbb R^o`"""

    input_dim: int
    output_dim: int

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        pass


class MLP(nn.Sequential, FFN):
    r"""An :class:`MLP` is an FFN that implements the following function:

    .. math::
        \mathbf h_0 &= \mathbf W_0 \mathbf x \,+ \mathbf b_{0} \\
        \mathbf h_l &= \mathbf W_l \left( \mathtt{dropout} \left( \sigma ( \,\mathbf h_{l-1}\, ) \right) \right) + \mathbf b_l\\

    where :math:`\mathbf x` is the input tensor, :math:`\mathbf W_l` and :math:`\mathbf b_l`
    are the learned weight matrix and bias, respectively, of the :math:`l`-th layer,
    :math:`\mathbf h_l` is the hidden representation after layer :math:`l`, and :math:`\sigma`
    is the activation function.
    """

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        dropout = nn.Dropout(dropout)
        act = get_activation_function(activation)
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        blocks = [nn.Sequential(nn.Linear(dims[0], dims[1]))]
        if len(dims) > 2:
            blocks.extend(
                [
                    nn.Sequential(act, dropout, nn.Linear(d1, d2))
                    for d1, d2 in zip(dims[1:-1], dims[2:])
                ]
            )

        return cls(*blocks)

    @property
    def input_dim(self) -> int:
        return self[0][-1].in_features

    @property
    def output_dim(self) -> int:
        return self[-1][-1].out_features
