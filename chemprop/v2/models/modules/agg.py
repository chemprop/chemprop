from abc import ABC, abstractmethod
from typing import Collection, Iterable

import torch
from torch import Tensor, nn

from chemprop.v2.utils.factory import ClassFactory

AggregationFactory = ClassFactory()


class Aggregation(ABC, nn.Module):
    """An `Aggregation` module aggregates inputs along the given dimension"""

    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, H_v: Tensor, sizes: Iterable[int]) -> Tensor:
        """Aggregate node-level representations into a graph-level representation

        The input `H_v` is a stacked tensor containing the node-level representations for
        `b` separate graphs. NOTE: it is possible for a graph to have 0 nodes. In this case, the
        representation will be a zero vector of length `d` in the final output.

        I.e., if `H_v` is a tensor of shape `10 x 4` and `sizes` is equal to `[3, 4, 3]`, then `
        [:3]`,` H[3:7]`, and `H[7:]` correspond to the node-level represenataions of the three
        stacked graphs. The output of this function will then be a tensor of shape `3 x 4`

        Parameters
        ----------
        H_v : Tensor
            A tensor of shape `sum(sizes) x d` containing the stacked node-level representations of
            `b` graphs
        sizes : Iterable[int]
            an iterable of length `b` containing the number of nodes in each of the `b` graphs,
            respectively. NOTE: `sum(sizes)` must be equal to `len(H_v)`

        Returns
        -------
        Tensor
            a tensor of shape `b x d` containing the graph-level representations of each graph

        Raises
        ------
        RuntimeError
            if `sum(sizes)` is not equal to `len(H_v)`
        """
        H_vs = H_v.split(sizes)
        hs = self.aggregate(H_vs)

        return torch.stack(hs)

    @abstractmethod
    def aggregate(self, Hs: Iterable[Tensor]) -> Collection[Tensor]:
        """Calculate the aggregated representation of each node-level representation `H`
        
        Parameters
        ----------
        Hs : Iterable[Tensor]
            An iterable containing `b` tensors of shape `... x d` corresponding the node-level
            representation of a given graph
        
        Returns
        -------
        Collection[Tensor]
            a collections of `b` tensors of shape `d` containing the global representation of each
            input graph
        """


@AggregationFactory.register("mean")
class MeanAggregation(Aggregation):
    """Take the mean node-level representation as the graph-level representation"""

    def aggregate(self, Hs: Iterable[Tensor]):
        return [H.mean(self.dim) if H.shape[0] > 0 else torch.zeros(H.shape[1]) for H in Hs]


@AggregationFactory.register("norm")
class NormAggregation(Aggregation):
    """Take the summed node-level representation divided by a normalization constant as the
    graph-level representation"""

    def __init__(self, *args, norm: float = 100, **kwargs):
        self.norm = norm
        super().__init__(*args, **kwargs)

    def aggregate(self, Hs: Iterable[Tensor]):
        return [
            H.sum(self.dim) / self.norm if H.shape[0] > 0 else torch.zeros(H.shape[1]) for H in Hs
        ]


@AggregationFactory.register("sum")
class SumAggregation(Aggregation):
    """Take the summed node-level representation as the graph-level representation"""

    def aggregate(self, Hs: Iterable[Tensor]):
        return [H.sum(self.dim) if H.shape[0] > 0 else torch.zeros(H.shape[1]) for H in Hs]

