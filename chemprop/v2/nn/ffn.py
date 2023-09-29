from itertools import chain
from typing import Protocol

from torch import nn, Tensor

from chemprop.v2.models.utils import get_activation_function


class _FFNProto(Protocol):
    input_dim: int
    output_dim: int

    def forward(self, X: Tensor) -> Tensor:
        pass


class FFN(nn.Module, _FFNProto):
    """A :class:`FFN` is a fully differentiable function that maps a tensor of shape ``N x d_i`` to a tensor of shape ``N x d_o``
    
    :inherited-members:
    """


class SimpleFFN(FFN):
    r"""A :class:`SimpleFFN` is a simple FFN that implements the following function:
     
    .. math::
        \mathbf H_0 &= \mathbf X\,\mathbf W_0 + \mathbf b_0 \\
        \mathbf H_l &= \mathtt{dropout} \left(
            \sigma \left(\,\mathbf H_{l-1}\,\mathbf W_l \right)
        \right) \\
        \mathbf H_L &= \mathbf H_{L-1} \mathbf W_L + \mathbf b_L,

    where :math:`\mathbf X` is the input tensor, :math:`\mathbf W_l` is the learned weight matrix
    for the :math:`l`-th layer, :math:`\mathbf b_l` is the bias vector for the :math:`l`-th layer,
    :math:`\mathbf H_l` is the hidden representation at layer :math:`l`, :math:`\sigma` is the
    activation function, and :math:`L` is the number of layers.

    :inherited-members:
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()

        dropout = nn.Dropout(dropout)
        act = get_activation_function(activation)

        dims = [input_dim, *([hidden_dim] * n_layers), output_dim]
        blocks = ((dropout, nn.Linear(d1, d2), act) for d1, d2 in zip(dims[:-1], dims[1:]))
        layers = list(chain(*blocks))

        self.ffn = nn.Sequential(*layers[1:-1])
        self.input_dim = self.ffn[0].in_features
        self.output_dim = self.ffn[-1].out_features

    def forward(self, X):
        return self.ffn(X)
