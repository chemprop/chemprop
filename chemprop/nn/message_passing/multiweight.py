import torch
from torch import Tensor, nn
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.nn.message_passing.base import _BondMessagePassingMixin, _MessagePassingBase


class MultiweightMessagePassing(_BondMessagePassingMixin, _MessagePassingBase):
    r"""A variant of BondMessagePassing where the hidden weight matrix (W_h)
    is untied across message passing steps (depth).

    Instead of reapplying the same matrix, a distinct W_h_i is learned for each iteration.
    """

    def __init__(self, *args, **kwargs):
        # 1. Run the base initialization, which will temporarily create a single W_h
        super().__init__(*args, **kwargs)

        # 2. Extract dimensions and bias from the temporarily created matrix
        d_h = self.W_h.in_features
        bias = self.W_h.bias is not None

        # 3. Overwrite W_h with a ModuleList of untied matrices.
        # The message passing loop runs (depth - 1) times, so we need (depth - 1) matrices.
        self.W_h = nn.ModuleList([
            nn.Linear(d_h, d_h, bias=bias) for _ in range(self.depth - 1)
        ])

        # LayerNorms for regularization
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_h)
            for _ in range(self.depth - 1)
        ])

    def setup(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        bias: bool = False,
    ):
        # Standard setup required by the base class.
        # The single W_h returned here is immediately overwritten by our __init__ above.
        W_i = nn.Linear(d_v + d_e, d_h, bias)
        W_h = nn.Linear(d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None

        return W_i, W_h, W_o, W_d

    def update(self, M_t: Tensor, H_0: Tensor, step: int) -> Tensor:
        """Calculate the updated hidden state using the step-specific weight matrix"""
        # Select the specific layernorm/weight matrix for this depth iteration
        M_norm = self.norms[step](M_t)
        H_t = self.W_h[step](M_norm)
        H_t = self.tau(H_0 + H_t)
        H_t = self.dropout(H_t)

        return H_t

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        bmsg = self.graph_transform(bmg)
        H_0 = self.initialize(bmsg)

        H = self.tau(H_0)

        # We replace the `for _ in range(1, self.depth)` with an enumerated loop
        # so we can pass the step index (0 to depth-2) to the update function
        for step in range(self.depth - 1):
            if self.undirected:
                H = (H + H[bmsg.rev_edge_index]) / 2

            M = self.message(H, bmsg)
            H = self.update(M, H_0, step)

        index_torch = bmsg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(len(bmsg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )
        return self.finalize(M, bmsg.V, V_d)
