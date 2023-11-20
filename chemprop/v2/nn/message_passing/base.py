from typing import Protocol

from torch import nn, Tensor

from chemprop.v2.data import BatchMolGraph
from chemprop.v2.utils.hparams import HasHParams


class MessagePassingProto(Protocol):
    input_dim: int
    output_dim: int

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMolGraph
            the batch of :class:`~chemprop.v2.featurizers.molgraph.MolGraph`s to encode
        V_d : Tensor | None, default=None
            an optional tensor of shape `V x d_vd` containing additional descriptors for each atom
            in the batch. These will be concatenated to the learned atomic descriptors and
            transformed before the readout phase. NOTE: recall that `V` is equal to `num_atoms + 1`,
            so if provided, this tensor must be 0-padded in the 0th row.

        Returns
        -------
        Tensor
            a tensor of shape `V x d_h` or `V x (d_h + d_vd)` containing the hidden representation
            of each vertex in the batch of graphs. The feature dimension depends on whether
            additional atom descriptors were provided
        """


class MessagePassingBlock(nn.Module, MessagePassingProto, HasHParams):
    """A :class:`MessagePassingBlock` is encodes a batch of molecular graphs using message passing
    to learn vertex-level hidden representations."""
