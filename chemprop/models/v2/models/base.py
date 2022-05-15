from torch import Tensor, nn

from chemprop.nn_utils import get_activation_function

from chemprop.models.v2.encoders import MPNEncoder


class MoleculeModel(nn.Module):
    def __init__(
        self,
        encoder: MPNEncoder,
        num_tasks: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()

        self.encoder = encoder
        self.num_tasks = num_tasks
        self.ffn = self.build_ffn(
            encoder.output_dim, num_tasks, ffn_hidden_dim, ffn_num_layers, dropout, activation
        )

    def build_ffn(
        self,
        d_i: int,
        d_o: int,
        d_h: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> nn.Sequential:
        dropout = nn.Dropout(dropout)
        activation = get_activation_function(activation)

        if n_layers == 0:
            layers = [dropout, nn.Linear(d_i, d_o)]
        else:
            layers = [dropout, nn.Linear(d_i, d_h)]
            for _ in range(1, n_layers):
                layers.extend([activation, dropout, nn.Linear(d_h, d_h)])
            layers.extend([activation, dropout, nn.Linear(d_h, d_o)])

        return nn.Sequential(*layers)

    def fingerprint(self, *args) -> Tensor:
        """Calculate the learned fingerprint for the input molecules/reactions"""
        return self.encoder(*args)

    def encoding(self, *args) -> Tensor:
        """Calculate the encoding ("hidden representation") for the input molecules/reactions"""
        return self.ffn[:-1](self.encoder(*args))

    def forward(self, *args) -> Tensor:
        """Generate predictions for the input batch.

        NOTE: the input signature to this function is the same as the input `MoleculeModel`'s
        `forward()` function
        """
        return self.ffn(self.encoder(*args))
