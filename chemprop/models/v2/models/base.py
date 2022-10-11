from abc import ABC, abstractmethod
from itertools import chain
from typing import Optional

import torch
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
        self.ffn = self.build_ffn(
            encoder.output_dim,
            n_tasks * self.n_targets,
            ffn_hidden_dim,
            ffn_num_layers,
            dropout,
            activation
        )

    @property
    def n_targets(self) -> int:
        """The number of targets to predict per task.
        
        NOTE: this is *not* related to the number of classes to predict.  It is used as a multiplier
        for the output dimension of the MPNN
        """
        return 1

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

    def fingerprint(self, *args, X_f: Optional[Tensor] = None) -> Tensor:
        """Calculate the learned fingerprint for the input molecules/reactions"""
        H = self.encoder(*args)
        if X_f is not None:
            H = torch.cat((H, X_f), 1)

        return H

    def encoding(self, *args, X_f: Optional[Tensor] = None) -> Tensor:
        """Calculate the encoding ("hidden representation") for the input molecules/reactions"""
        return self.ffn[:-1](self.fingerprint(*args, X_f))

    def forward(self, *args, X_f: Optional[Tensor] = None) -> Tensor:
        """Generate predictions for the input batch.

        NOTE: the signature of `*args` the underlying `encoder.forward()`
        """
        return self.ffn(self.fingerprint(*args, X_f))

