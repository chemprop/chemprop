from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import Tensor, nn

from chemprop.v2.utils import HasHParams, ClassRegistry

AggregationRegistry = ClassRegistry()


class Aggregation(ABC, nn.Module, HasHParams):
    """An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
    a batch of graph-level representations

    **NOTE**: this class is abstract and cannot be instantiated. Instead, you must use one of the
    concrete subclasses.

    See also
    --------
    :class:`chemprop.v2.models.modules.agg.MeanAggregation`
    :class:`chemprop.v2.models.modules.agg.SumAggregation`
    :class:`chemprop.v2.models.modules.agg.NormAggregation`
    """

    def __init__(self, dim: int = 0):
        super().__init__()

        self.dim = dim
        self.hparams = {"dim": dim, "cls": self.__class__}

    def forward(self, H: Tensor, sizes: Sequence[int] | None) -> Tensor:
        """Aggregate the graph-level representations of a batch of graphs into their respective
        global representations

        NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
        a zero vector of length `d` in the final output.

        Parameters
        ----------
        H : Tensor
            A tensor of shape ``sum(sizes) x d`` containing the stacked node-level representations
            of ``len(sizes)`` graphs
        sizes : Sequence[int]
            an list containing the number of nodes in each graph, respectively.

        Returns
        -------
        Tensor
            a tensor of shape ``len(sizes) x d`` containing the graph-level representation of each
            graph

        Raises
        ------
        ValueError
            if ``sum(sizes)`` is not equal to ``len(H_v)``

        Examples
        --------
        **NOTE**: the following examples are for illustrative purposes only. In practice, you must
        use one of the concrete subclasses.

        1. A typical use-case:

        >>> H = torch.rand(10, 4)
        >>> sizes = [3, 4, 3]
        >>> agg = Aggregation()
        >>> agg(H, sizes).shape
        torch.Size([3, 4])

        2. A batch containing a graph with 0 nodes:

        >>> H = torch.rand(10, 4)
        >>> sizes = [3, 4, 0, 3]
        >>> agg = Aggregation()
        >>> agg(H, sizes).shape
        torch.Size([4, 4])
        """
        try:
            hs = [
                self.agg(H_i) if len(H_i) > 0 else torch.zeros(H.shape[1]) for H_i in H.split(sizes)
            ]
        except RuntimeError:
            raise ValueError(f"arg 'sizes' must sum to `len(H)` ({len(H)})! got: {sum(sizes)}")

        return torch.stack(hs)

    @abstractmethod
    def agg(self, H: Tensor) -> Tensor:
        r"""Aggregate the graph-level of a single graph into a vector

        Parameters
        ----------
        H : Tensor
            A tensor of shape ``V x d`` containing the node-level representation of a graph with
            ``V`` nodes and node feature dimension ``d``

        Returns
        -------
        Tensor
            a tensor of shape ``d`` containing the global representation of the input graph
        """


@AggregationRegistry.register("mean")
class MeanAggregation(Aggregation):
    r"""Average the graph-level representation

    .. math::
        \mathbf h = \frac{1}{|V|} \sum_{v \in V} \mathbf h_v"""

    def agg(self, H: Tensor) -> Tensor:
        return H.mean(self.dim)


@AggregationRegistry.register("sum")
class SumAggregation(Aggregation):
    r"""Sum the graph-level representation

    .. math::
        \mathbf h = \sum_{v \in V} \mathbf h_v

    """

    def agg(self, H: Tensor) -> Tensor:
        return H.sum(self.dim)


@AggregationRegistry.register("norm")
class NormAggregation(Aggregation):
    r"""Sum the graph-level representation and divide by a normalization constant

    .. math::
        \mathbf h = \frac{1}{c} \sum_{v \in V} \mathbf h_v
    """

    def __init__(self, *args, norm: float = 100, **kwargs):
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.hparams["norm"] = norm

    def agg(self, H: Tensor) -> Tensor:
        return H.sum(self.dim) / self.norm
