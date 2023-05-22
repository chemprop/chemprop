from itertools import chain

from torch import nn

from chemprop.v2.models.utils import get_activation_function


class FFN(nn.Module):
    input_dim: int
    output_dim: int


def build_ffn(
    input_dim: int,
    n_tasks: int,
    n_targets: int,
    hidden_dim: int = 300,
    n_layers: int = 1,
    dropout: float = 0.0,
    activation: str = "relu",
) -> nn.Sequential:
    output_dim = n_tasks * n_targets
    dropout = nn.Dropout(dropout)
    act = get_activation_function(activation)

    sizes = [input_dim, *([hidden_dim] * n_layers), output_dim]
    blocks = ((dropout, nn.Linear(d1, d2), act) for d1, d2 in zip(sizes[:-1], sizes[1:]))
    layers = list(chain(*blocks))

    ffn = nn.Sequential(*layers[:-1])
    ffn.input_dim = input_dim
    ffn.output_dim = output_dim

    return ffn