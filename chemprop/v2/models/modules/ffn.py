from itertools import chain
from typing import Protocol

from torch import nn

from chemprop.v2.models.utils import get_activation_function


class FFNProto(Protocol):
    input_dim: int
    output_dim: int


class FFN(nn.Module, FFNProto):
    pass


class SimpleFFN(FFN):
    def __init__(
        self,
        input_dim: int,
        n_tasks: int,
        n_targets: int,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()

        input_dim = input_dim
        output_dim = n_tasks * n_targets
        dropout = nn.Dropout(dropout)
        act = get_activation_function(activation)

        sizes = [input_dim, *([hidden_dim] * n_layers), output_dim]
        blocks = ((dropout, nn.Linear(d1, d2), act) for d1, d2 in zip(sizes[:-1], sizes[1:]))
        layers = list(chain(*blocks))

        self.ffn = nn.Sequential(*layers[1:-1])
        self.input_dim = self.ffn[0].in_features
        self.output_dim = self.ffn[-1].out_features

    def forward(self, X):
        return self.ffn(X)