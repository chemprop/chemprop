from itertools import chain
from typing import Optional, Union

import torch
from torch import Tensor, nn

from chemprop.nn_utils import get_activation_function
from chemprop.models.v2.modules import MessagePassingBlock, MolecularInput, ReactionInput


class MPNN(nn.Module):
    """An `MPNN` is comprised of a `MessagePassingBlock` and an FFN top-model. The former
    calculates learned encodings from an input molecule/reaction graph, and the latter takes these
    encodings as input to calculate a final prediction. The full model is trained end-to-end.

    An `MPNN` takes a input a molecular graph and outputs a tensor of shape `b x t * s`, where `b`
    the size of the batch (i.e., number of molecules in the graph,) `t` is the number of tasks to
    predict, and `s` is the number of targets to predict per task.

    NOTE: the number of targets `s` is *not* related to the number of classes to predict.  It is
    used as a multiplier for the output dimension of the MPNN when the predictions correspond to a
    parameterized distribution, e.g., MVE regression, for which `s` is 2. Typically, this is just 1.
    """
    def __init__(
        self,
        encoder: MessagePassingBlock,
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

    def fingerprint(
        self, inputs: Union[MolecularInput, ReactionInput], X_f: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the learned fingerprint for the input molecules/reactions"""
        H = self.encoder(*inputs)
        if X_f is not None:
            H = torch.cat((H, X_f), 1)

        return H

    def encoding(
        self, inputs: Union[MolecularInput, ReactionInput], X_f: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the encoding ("hidden representation") for the input molecules/reactions"""
        return self.ffn[:-1](self.fingerprint(*inputs, X_f))

    def forward(
        self, inputs: Union[MolecularInput, ReactionInput], X_f: Optional[Tensor] = None
    ) -> Tensor:
        """Generate predictions for the input batch.

        NOTE: the type signature of `input` matches the underlying `encoder.forward()`
        """
        return self.ffn(self.fingerprint(inputs, X_f))

