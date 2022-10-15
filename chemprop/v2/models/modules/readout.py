from abc import ABC, abstractmethod
from typing import Iterable

import torch
from torch import Tensor, nn

from chemprop.v2.utils import RegistryMixin


class Readout(ABC, nn.Module, RegistryMixin):
    """An `Readout` module aggregates the node-level representations of graph into a single
    graph-level representation"""

    registry = {}

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, H_v: Tensor, sizes: Iterable[int]) -> Tensor:
        """Aggregate node-level representations into a graph-level representation

        The input `H_v` is a stacked tensor containing the node-level representations for
        `b` separate graphs. I.e., if `H_v` is a tensor of shape `10 x 4` and `sizes` is equal to
        `[3, 4, 3]`, then `H[:3]`,` H[3:7]`, and `H[7:]` correspond to the node-level
        represenataions of the three stacked graphs. The output of this function will then be a
        tensor of shape `3 x 4`

        NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
        a zero vector of length `d` in the final output.

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
    def aggregate(self, Hs: Iterable[Tensor]):
        pass


class MeanReadout(Readout):
    """Take the mean node-level representation as the graph-level representation"""

    alias = "mean"

    def aggregate(self, Hs: Iterable[Tensor]):
        return [H.mean(0) if H.shape[0] > 0 else torch.zeros(H.shape[1]) for H in Hs]


class NormReadout(Readout):
    """Take the summed node-level representation divided by a normalization constant as the
    graph-level representation"""

    alias = "norm"

    def __init__(self, *args, norm: float = 100, **kwargs):
        self.norm = norm
        super().__init__(*args, **kwargs)

    def aggregate(self, Hs: Iterable[Tensor]):
        return [H.sum(0) / self.norm if H.shape[0] > 0 else torch.zeros(H.shape[1]) for H in Hs]


class SumReadout(Readout):
    """Take the summed node-level representation as the graph-level representation"""

    alias = "sum"

    def aggregate(self, Hs: Iterable[Tensor]):
        return [H.sum(0) if H.shape[0] > 0 else torch.zeros(H.shape[1]) for H in Hs]


def build_readout(aggregation: str = "mean", norm: float = 100) -> Readout:
    try:
        aggr_cls = Readout.registry[aggregation.lower()]
    except KeyError:
        raise ValueError(
            f"Invalid aggregation! got: '{aggregation}'. "
            f"expected one of {set(Readout.registry.keys())}"
        )

    return aggr_cls(norm)
