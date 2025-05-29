from abc import abstractmethod

from torch import Tensor, nn

from chemprop.data import BatchMolGraph
from chemprop.nn.hparams import HasHParams


class MessagePassing(nn.Module, HasHParams):
    """A :class:`MessagePassing` module encodes a batch of molecular graphs
    using message passing to learn vertex-level hidden representations."""

    output_dim: int

    @abstractmethod
    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMolGraph
            the batch of :class:`~chemprop.featurizers.molgraph.MolGraph`\s to encode
        V_d : Tensor | None, default=None
            an optional tensor of shape `V x d_vd` containing additional descriptors for each vertex
            in the batch. These will be concatenated to the learned vertex descriptors and
            transformed before the readout phase.

        Returns
        -------
        Tensor
            a tensor of shape `V x d_h` or `V x (d_h + d_vd)` containing the hidden representation
            of each vertex in the batch of graphs. The feature dimension depends on whether
            additional vertex descriptors were provided
        """


class MABMessagePassing(nn.Module, HasHParams):
    """A :class:`MABMessagePassing` module encodes a batch of molecular graphs
    using message passing to learn both vertex-level and edge-level hidden representations."""

    output_dims: tuple[int | None, int | None]

    @abstractmethod
    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, E_d: Tensor | None = None
    ) -> tuple[Tensor | None, Tensor | None]:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMolGraph
            the batch of :class:`~chemprop.featurizers.molgraph.MolGraph`\s to encode
        V_d : Tensor | None, default=None
            an optional tensor of shape `V x d_vd` containing additional descriptors for each vertex
            in the batch. These will be concatenated to the learned vertex descriptors and
            transformed before the readout phase.
        E_d : Tensor | None, default=None
            an optional tensor of shape `E x d_ed` containing additional descriptors for each
            directed edge in the batch. These will be concatenated to the learned edge descriptors
            and transformed before the readout phase. NOTE: There are two directed edges per graph
            connection. If the extra descriptors are for the connections, each row should be
            repeated twice in the tensor, once for each direction, potentially using
            ``E_d = np.repeat(E_d, repeats=2, axis=0)``.

        Returns
        -------
        tuple[Tensor | None, Tensor | None]
            Two tensors of shape `V x d_h` or `V x (d_h + d_vd)` and `E x dh` or `E x (dh + d_ed)`
            containing the hidden representation of each vertex and edge in the batch of graphs.
            The feature dimension depends on whether additional atom/bond descriptors were provided.
            If either the vertex or edge hidden representations are not needed, computing the
            corresponding tensor can be suppresed by setting either return_vertex_embeddings or
            return_edge_embeddings to `False` when initializing the module.
        """
