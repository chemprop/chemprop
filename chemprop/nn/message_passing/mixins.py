import torch
from torch import Tensor

from chemprop.data import BatchMolGraph


class _BondMessagePassingMixin:
    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return self.W_i(torch.cat([bmg.V[bmg.edge_index[0]], bmg.E], dim=1))

    def message(self, H: Tensor, bmg: BatchMolGraph) -> Tensor:
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )[bmg.edge_index[0]]
        M_rev = H[bmg.rev_edge_index]

        return M_all - M_rev


class _AtomMessagePassingMixin:
    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return self.W_i(bmg.V[bmg.edge_index[0]])

    def message(self, H: Tensor, bmg: BatchMolGraph):
        H = torch.cat((H, bmg.E), dim=1)
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        return torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )[bmg.edge_index[0]]
