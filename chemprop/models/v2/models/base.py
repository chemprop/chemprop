from itertools import chain
from torch import Tensor, nn

from chemprop.nn_utils import get_activation_function
from chemprop.models.v2.encoders import MPNEncoder


class MPNN(nn.Module):
    def __init__(
        self,
        encoder: MPNEncoder,
        n_tasks: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()

        self.encoder = encoder
        self.n_tasks = n_tasks
        self.n_targets = n_tasks
        self.ffn = self.build_ffn(
            encoder.output_dim, n_tasks, ffn_hidden_dim, ffn_num_layers, dropout, activation
        )

    @staticmethod
    def build_ffn(
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> nn.Sequential:
        dropout = nn.Dropout(dropout)
        activation = get_activation_function(activation)
        layers = [hidden_dim] * n_layers

        layers = [input_dim, *layers, output_dim]
        ffn = list(
            chain(
                *((dropout, nn.Linear(d1, d2), activation)
                for d1, d2 in zip(layers[:-1], layers[1:]))
            )
        )

        return nn.Sequential(*ffn[:-1])

    def fingerprint(self, *args) -> Tensor:
        """Calculate the learned fingerprint for the input molecules/reactions"""
        return self.encoder(*args)

    def encoding(self, *args) -> Tensor:
        """Calculate the encoding ("hidden representation") for the input molecules/reactions"""
        return self.ffn[:-1](self.encoder(*args))

    def forward(self, *args) -> Tensor:
        """Generate predictions for the input batch.

        NOTE: the input signature to this function matches the underlying `encoder.forward()`
        """
        return self.ffn(self.encoder(*args))
