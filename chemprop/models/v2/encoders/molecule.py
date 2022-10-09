from __future__ import annotations

from abc import abstractmethod
from ctypes import Union
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
from chemprop.featurizers.v2.molgraph import BatchMolGraph

from chemprop.models.v2.encoders.base import MPNEncoder
from chemprop.models.v2.encoders.utils import Aggregation
from chemprop.nn_utils import get_activation_function


MoleculeEncoderInput = tuple[BatchMolGraph, Optional[Tensor]]


class MoleculeEncoder(MPNEncoder):
    def __init__(
        self,
        d_v: int,
        d_e: int,
        d_h: int = 300,
        bias: bool = False,
        depth: int = 3,
        undirected: bool = False,
        # layers_per_message: int,
        dropout: float = 0,
        activation: str = "relu",
        aggregation: Union[str, Aggregation] = Aggregation.MEAN,
        atom_descriptors: Optional[str] = None,
        d_vd: Optional[int] = None,
    ):
        super().__init__()

        self.depth = depth
        self.undirected = undirected
        # self.layers_per_message = 1

        self.dropout = nn.Dropout(dropout)
        self.act = get_activation_function(activation)
        self.agg = Aggregation.get(aggregation) if isinstance(aggregation, str) else aggregation

        self.cached_zero_vector = nn.Parameter(torch.zeros(d_h), requires_grad=False)

        self.__output_dim = d_h

        if atom_descriptors == "descriptor":
            self.d_vd = d_vd
            self.__output_dim += d_vd
            self.fc_vd = nn.Linear(d_h + d_vd, d_h + d_vd)

        self.setup_weight_matrices(d_v, d_e, d_h, bias)

    @abstractmethod
    def setup_weight_matrices(self, d_v: int, d_e: int, d_h: int = 300, bias: bool = False):
        """set up the weight matrices used in the message passing udpate functions
        
        Parameters
        ----------
        d_v : int
            the input vertex feature dimension
        d_e : int
            the input edge feature dimension
        d_h : int
            the hidden dimension during message passing
        bias: bool
            whether to add a learned bias to the matrices
        """

    @property
    def output_dim(self) -> int:
        return self.__output_dim

    def cat_descriptors(self, H_v: Tensor, X_vd: Tensor) -> Tensor:
        """Concatenate the atom descriptors `X_vd` onto the hidden representations `H_v`

        Parameters
        ----------
        H_v : Tensor
            a tensor of shape `V x d_h` containing the hidden representation of each atom
        X_vd : Tensor
            a tensor of shape `V x d_vd` containing additional descriptors for each atom

        Returns
        -------
        Tensor
            a tensor of shape `V x (d_h + d_vd)` containing the transformed hidden representations

        Raises
        ------
        ValueError
            if `X_vd` is of incorrect shape
        """
        try:
            H_vd = torch.cat((H_v, X_vd), 1)
            H_v = self.fc_vd(H_vd)
        except RuntimeError:
            raise ValueError(
                "arg 'X_vd' has incorrect shape! "
                f"got: `{' x '.join(map(str, X_vd.shape))}`. expected: `{len(H_v)} x {self.d_vd}`"
            )

        H_v = self.dropout(H_v)

        return H_v  # V x (d_h + d_vd)

    def readout(self, H_v: Tensor, sizes: Iterable[int]) -> Tensor:
        h_vs = torch.split(H_v, sizes)
        hs = [h_v.sum(0) if size > 0 else self.cached_zero_vector for h_v, size in zip(h_vs, sizes)]

        if self.agg == Aggregation.MEAN:
            hs = [h / size if size > 0 else h for h, size in zip(hs, sizes)]
        elif self.agg == Aggregation.NORM:
            hs = [h / self.aggregation_norm if size > 0 else h for h, size in zip(hs, sizes)]
        elif self.agg == Aggregation.SUM:
            pass
        else:
            raise RuntimeError

        return torch.stack(hs, 0)

    @abstractmethod
    def forward(self, bmg: BatchMolGraph, X_vd: Optional[Tensor] = None) -> Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMolGraph
            the batch of `MolGraphs` to encode
        X_vd : Optional[Tensor]
            an optional tensor of shape `V x d_vd` containing additional descriptors for each atom
            in the batch. These will be concatenated to the learned atomic descriptors and
            transformed before the readout phase. NOTE: recall that `V` is equal to `num_atoms` + 1,
            so if provided, this tensor must be 0-padded in the 0th row.

        Returns
        -------
        Tensor
            a tensor of shape `B x d_h` or `B x (d_h + d_vd)` containing the encoding of each
            molecule in the batch, depending on whether additional atom descriptors were provided
        """


class BondMessageEncoder(MoleculeEncoder):
    def setup_weight_matrices(self, d_v: int, d_e: int, d_h: int = 300, bias: bool = False):
        self.W_i = nn.Linear(d_e, d_h, bias)
        self.W_h = nn.Linear(d_h, d_h, bias)
        self.W_o = nn.Linear(d_v + d_h, d_h)

    def forward(self, bmg: BatchMolGraph, X_vd: Optional[Tensor] = None) -> Tensor:
        H_0 = self.W_i(bmg.X_e)  # E x d_h

        H_e = self.act(H_0)
        for _ in range(1, self.depth):
            if self.undirected:
                H_e = (H_e + H_e[bmg.b2revb]) / 2

            M_e = H_e[bmg.a2b].sum(1)[bmg.b2a] - H_e[bmg.b2revb]  # E x d_h

            H_e = H_0 + self.W_h(M_e)  # E x d_h
            H_e = self.act(H_e)
            H_e = self.dropout(H_e)

        M_v = H_e[bmg.a2b].sum(1)  # V x d_h
        H_v = self.W_o(torch.cat((bmg.X_v, M_v), 1))  # V x d_h
        H_v = self.act(H_v)
        H_v = self.dropout(H_v)

        if X_vd is not None:
            H_v = self.cat_descriptors(H_v, X_vd)

        H = self.readout(H_v[1:], [n_a for _, n_a in bmg.a_scope])  # B x d_h + (d_vd)

        return H


class AtomMessageEncoder(MoleculeEncoder):
    def setup_weight_matrices(self, d_v: int, d_e: int, d_h: int = 300, bias: bool = False):
        self.W_i = nn.Linear(d_v, d_h, bias)
        self.W_h = nn.Linear(d_e + d_h, d_h, bias)
        self.W_o = nn.Linear(d_v + d_h, d_h)

    def forward(self, bmg: BatchMolGraph, X_vd: Optional[Tensor] = None) -> Tensor:
        H_0 = self.W_i(bmg.X_v)  # V x d_h
        H_v = self.act(H_0)

        for _ in range(1, self.depth):
            if self.undirected:
                H_v = (H_v + H_v[bmg.b2revb]) / 2

            M_v_k = torch.cat((H_v[bmg.a2a], bmg.X_e[bmg.a2b]), 2)  # V x b x (d_h + d_e)
            M_v = M_v_k.sum(1)  # V x d_h + d_e

            H_v = H_0 + self.W_h(M_v)  # E x d_h
            H_v = self.act(H_v)
            H_v = self.dropout(H_v)

        M_v_k = H_v[bmg.a2a]
        M_v = M_v_k.sum(1)  # V x d_h

        H_v = self.act(self.W_o(torch.cat((bmg.X_v, M_v), 1)))  # V x d_h
        H_v = self.dropout(H_v)

        if X_vd is not None:
            H_v = self.cat_descriptors(H_v, X_vd)

        H = self.readout(H_v[1:], [n_a for _, n_a in bmg.a_scope])  # B x d_h (+ d_vd)

        return H


def molecule_encoder(d_v: int, d_e: int, bond_messages: bool = True, *args, **kwargs):
    if bond_messages:
        encoder = BondMessageEncoder(d_v, d_e, *args, **kwargs)
    else:
        encoder = AtomMessageEncoder(d_v, d_e, *args, **kwargs)

    return encoder
