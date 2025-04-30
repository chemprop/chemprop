from abc import abstractmethod

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn

from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.exceptions import InvalidShapeError
from chemprop.nn.message_passing.mixins import _AtomMessagePassingMixin, _BondMessagePassingMixin
from chemprop.nn.message_passing.proto import MABMessagePassing
from chemprop.nn.transforms import GraphTransform, ScaleTransform
from chemprop.nn.utils import Activation, get_activation_function
from chemprop.nn.message_passing import (
    _MessagePassingBase,
    _BondMessagePassingMixin,
    _AtomMessagePassingMixin,
)


class _MixedMessagePassingBase(_MessagePassingBase, MixedMessagePassing):
    @property
    def output_dims(self) -> tuple[int | None, int | None]:
        """Returns the output dimensions of the vertex and edge embeddings."""
        return (
            self.W_d.out_features if self.W_d is not None else self.W_o.out_features,
            self.W_ed.out_features if self.W_ed is not None else self.W_o_b.out_features,
        )


class MixedBondMessagePassing(_MixedMessagePassingBase, _BondMessagePassingMixin):
    def setup(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        d_ed: int | None = None,
        bias: bool = False,
    ):
        W_i = nn.Linear(d_v + d_e, d_h, bias)
        W_h = nn.Linear(d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        W_o_b = nn.Linear(d_e + d_h, d_h)
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None
        W_ed = nn.Linear(d_h + d_ed, d_h + d_ed) if d_ed else None

        return W_i, W_h, W_o, W_d, W_o_b, W_ed

    def finalize(
        self, H: Tensor, M: Tensor, V: Tensor, E: Tensor, V_d: Tensor | None, E_d: Tensor | None
    ) -> tuple[Tensor]:
        H_v = self.W_o(torch.cat((V, M), dim=1))
        H_v = self.tau(H_v)
        H_v = self.dropout(H_v)
        H_b = self.W_o_b(torch.cat((E, H), dim=1))
        H_b = self.tau(H_b)
        H_b = self.dropout(H_b)

        if V_d is not None:
            V_d = self.V_d_transform(V_d)
            try:
                H_v = self.W_d(torch.cat((H_v, V_d), dim=1))  # V x (d_o + d_vd)
                H_v = self.dropout(H_v)
            except RuntimeError:
                raise InvalidShapeError("V_d", V_d.shape, [len(H_v), self.W_d.in_features])

        if E_d is not None:
            E_d = self.E_d_transform(E_d)
            try:
                H_b = self.W_ed(torch.cat((H_b, E_d), dim=1))
                H_b = self.dropout(H_b)
            except RuntimeError:
                raise InvalidShapeError("E_d", E_d.shape, [len(H_b), self.W_ed.in_features])

        return H_v, H_b

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, E_d: Tensor | None = None
    ) -> tuple[Tensor | None, Tensor | None]:
        bmg = self.graph_transform(bmg)
        H_0 = self.initialize(bmg)

        H = self.tau(H_0)
        for _ in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2

            M = self.message(H, bmg)
            H = self.update(M, H_0)

        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )
        return self.finalize(H, M, bmg.V, bmg.E, V_d, E_d)

    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return _BondMessagePassingMixin.initialize(self, bmg)

    def message(self, H: Tensor, bmg: BatchMolGraph) -> Tensor:
        return _BondMessagePassingMixin.message(self, H, bmg)


class MixedAtomMessagePassing(_MixedMessagePassingBase, _AtomMessagePassingMixin):
    def setup(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        d_ed: int | None = None,
        bias: bool = False,
    ):
        W_i = nn.Linear(d_v, d_h, bias)
        W_h = nn.Linear(d_e + d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        W_o_b = nn.Linear(d_e + d_h, d_h)
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None
        W_ed = nn.Linear(d_h + d_ed, d_h + d_ed) if d_ed else None

        return W_i, W_h, W_o, W_d, W_o_b, W_ed

    def finalize(
        self, H: Tensor, M: Tensor, V: Tensor, E: Tensor, V_d: Tensor | None, E_d: Tensor | None
    ) -> tuple[Tensor]:
        H_v = self.W_o(torch.cat((V, M), dim=1))
        H_v = self.tau(H_v)
        H_v = self.dropout(H_v)
        H_b = self.W_o_b(torch.cat((E, H), dim=1))
        H_b = self.tau(H_b)
        H_b = self.dropout(H_b)

        if V_d is not None:
            V_d = self.V_d_transform(V_d)
            try:
                H_v = self.W_d(torch.cat((H_v, V_d), dim=1))  # V x (d_o + d_vd)
                H_v = self.dropout(H_v)
            except RuntimeError:
                raise InvalidShapeError("V_d", V_d.shape, [len(H_v), self.W_d.in_features])

        if E_d is not None:
            E_d = self.E_d_transform(E_d)
            try:
                H_b = self.W_ed(torch.cat((H_b, V_d), dim=1))
                H_b = self.dropout(H_b)
            except RuntimeError:
                raise InvalidShapeError("E_d", E_d.shape, [len(H_b), self.W_ed.in_features])

        return H_v, H_b

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, E_d: Tensor | None = None
    ) -> tuple[Tensor]:
        bmg = self.graph_transform(bmg)
        H_0 = self.initialize(bmg)

        H = self.tau(H_0)
        for _ in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2

            M = self.message(H, bmg)
            H = self.update(M, H_0)

        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )
        return self.finalize(H, M, bmg.V, bmg.E, V_d, E_d)

    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return _AtomMessagePassingMixin.initialize(self, bmg)

    def message(self, H: Tensor, bmg: BatchMolGraph):
        return _AtomMessagePassingMixin.message(self, H, bmg)
