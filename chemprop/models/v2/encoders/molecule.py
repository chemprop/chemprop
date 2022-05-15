from typing import Iterable, Optional, Tuple

from torch import Tensor
import torch

from chemprop.models.v2.encoders.base import MoleculeEncoder


class BondMessageEncoder(MoleculeEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(True, *args, **kwargs)

    def forward(
        self,
        X_v: Tensor,
        X_e: Tensor,
        a2b: Tensor,
        b2a: Tensor,
        b2revb: Tensor,
        a_scope: Iterable[Tuple],
        b_scope: Optional[Iterable[Tuple]] = None,
        a2a: Optional[Tensor] = None,
        X_v_d: Optional[Tensor] = None,
    ) -> Tensor:
        H_0 = self.W_i(X_e)  # E x d_h

        H_e = self.act(H_0)
        for _ in range(1, self.depth):
            if self.undirected:
                H_e = (H_e + H_e[b2revb]) / 2

            M_e = H_e[a2b].sum(1)[b2a] - H_e[b2revb]  # E x d_h

            H_e = H_0 + self.W_h(M_e)  # E x d_h
            H_e = self.act(H_e)
            H_e = self.dropout(H_e)

        M_v = H_e[a2b].sum(1)  # V x d_h
        H_v = self.W_o(torch.cat((X_v, M_v), 1))  # V x d_h
        H_v = self.act(H_v)
        H_v = self.dropout(H_v)

        if X_v_d is not None:
            H_v = self.concatenate_descriptors(H_v, X_v_d)

        H = self.readout(H_v[1:], [n_a for _, n_a in a_scope])  # B x d_h OR B x (d_h + d_vd)

        return H


class AtomMessageEncoder(MoleculeEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(False, *args, **kwargs)

    def forward(
        self,
        X_v: Tensor,
        X_e: Tensor,
        a2b: Tensor,
        b2a: Tensor,
        b2revb: Tensor,
        a_scope: Iterable[Tuple],
        b_scope: Optional[Iterable[Tuple]],
        a2a: Tensor = None,
        X_v_d: Optional[Tensor] = None,
    ) -> Tensor:
        H_0 = self.W_i(X_v)  # V x d_h
        H_v = self.act(H_0)

        for _ in range(1, self.depth):
            if self.undirected:
                H_v = (H_v + H_v[b2revb]) / 2

            M_v_k = torch.cat((H_v[a2a], X_e[a2b]), 2)  # V x b x (d_h + d_e)
            M_v = M_v_k.sum(1)  # V x d_h + d_e

            H_v = H_0 + self.W_h(M_v)  # E x d_h
            H_v = self.act(H_v)
            H_v = self.dropout(H_v)

        M_v_k = H_v[a2a]
        M_v = M_v_k.sum(1)  # V x d_h

        H_v = self.act(self.W_o(torch.cat((X_v, M_v), 1)))  # V x d_h
        H_v = self.dropout(H_v)

        if X_v_d is not None:
            H_v = self.concatenate_descriptors(H_v, X_v_d)

        H = self.readout(H_v[1:], [n_a for _, n_a in a_scope])  # B x d_h OR B x (d_h + d_vd)

        return H


def build_molecule_encoder(bond_messages: bool, *args, **kwargs):
    if bond_messages:
        return BondMessageEncoder(*args, **kwargs)

    return AtomMessageEncoder(*args, **kwargs)
