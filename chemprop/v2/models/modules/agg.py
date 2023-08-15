from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import Tensor, nn

from chemprop.v2.utils import ClassRegistry, pretty_shape

AggregationRegistry = ClassRegistry()


class Aggregation(ABC, nn.Module):
    """An `Aggregation` module aggregates aggregates the graph-level representation into a global
    representation"""

    def __init__(self, dim: int = 0):
        super().__init__()

        self.dim = dim

    def forward(self, H: Tensor, sizes: Sequence[int] | None) -> Tensor:
        """Aggregate the graph-level representations of a batch of graphs into their respective
        global representations

        NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
        a zero vector of length `d` in the final output.

        E.g., `H` is a tensor of shape `10 x 4` and `sizes` is equal to `[3, 4, 3]`, then
        `H[:3]`, `H[3:7]`, and `H[7:]` correspond to the graph-level represenataions of the three
        individual graphs. The output of a call to `forward()` will be a tensor of shape `3 x 4`

        Parameters
        ----------
        H : Tensor
            A tensor of shape `sum(sizes) x d` containing the stacked graph-level representations of
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
        ValueError
            if `sum(sizes)` is not equal to `len(H_v)`
        """
        try:
            hs = [
                self.agg(H_i) if len(H_i) > 0 else torch.zeros(H.shape[1]) for H_i in H.split(sizes)
            ]
        except RuntimeError:
            raise ValueError(
                f"arg 'sizes' must sum to `len(H)`! "
                f"got: {sum(sizes)} ({sizes}) but 'H' has shape {pretty_shape(H.shape)}"
            )

        return torch.stack(hs)

    @abstractmethod
    def agg(self, H: Tensor) -> Tensor:
        """Aggregate the graph-level of a single graph into a vector
        
        Parameters
        ----------
        H : Tensor
            A tensor of shape `V x d` containing the node-level representation of a graph with
            `V` nodes and node feature dimension `d`
        
        Returns
        -------
        Tensor
            a tensor of shape `d` containing the global representation of the input graph
        """


@AggregationRegistry.register("mean")
class MeanAggregation(Aggregation):
    """Average the graph-level representation"""

    def agg(self, H: Tensor) -> Tensor:
        return H.mean(self.dim)


@AggregationRegistry.register("sum")
class SumAggregation(Aggregation):
    """Sum the graph-level representation"""

    def agg(self, H: Tensor) -> Tensor:
        return H.sum(self.dim)


@AggregationRegistry.register("norm")
class NormAggregation(Aggregation):
    """Sum the graph-level representation and divide by a normalization constant"""

    def __init__(self, *args, norm: float = 100, **kwargs):
        super().__init__(*args, **kwargs)

        self.norm = norm

    def agg(self, H: Tensor) -> Tensor:
        return H.sum(self.dim) / self.norm



