from abc import abstractmethod

from torch import Tensor, nn

from chemprop.data import BatchMolGraph
from chemprop.nn.hparams import HasHParams


class MessagePassing(nn.Module, HasHParams):
    """A :class:`MessagePassing` module encodes a batch of molecular graphs
    using message passing to learn vertex-level hidden representations."""

    input_dim: int
    output_dim: int

    @abstractmethod
    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMolGraph
            the batch of :class:`~chemprop.featurizers.molgraph.MolGraph`\s to encode
        V_d : Tensor | None, default=None
            an optional tensor of shape `V x d_vd` containing additional descriptors for each atom
            in the batch. These will be concatenated to the learned atomic descriptors and
            transformed before the readout phase.

        Returns
        -------
        Tensor
            a tensor of shape `V x d_h` or `V x (d_h + d_vd)` containing the hidden representation
            of each vertex in the batch of graphs. The feature dimension depends on whether
            additional atom descriptors were provided
        """
