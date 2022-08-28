from __future__ import annotations

import warnings
from typing import Iterable

import torch
from torch import Tensor, nn

from chemprop.models.v2.encoders.base import MPNEncoder
from chemprop.models.v2.encoders.molecule import (
    MoleculeEncoder,
    MoleculeEncoderInput,
    molecule_encoder,
)


class ReactionEncoder(MPNEncoder):
    def __init__(self, encoders: Iterable[MoleculeEncoder], n_mols: int, shared: bool = False):
        super().__init__()

        if len(encoders) == 0:
            raise ValueError("arg `encoders` was empty!")

        if shared:
            if len(encoders) > 1:
                warnings.warn(
                    "More than 1 encoder was supplied but `shared` was True! "
                    "Using only the 0th encoder."
                )
        elif len(encoders) != n_mols:
            raise ValueError(
                "arg `n_mols` must be equal to `len(encoders)` if `shared` is False! "
                f"got: {n_mols} and {len(encoders)}, respectively."
            )

        self.n_mols = n_mols
        self.shared = shared

        if self.shared:
            self.encoders = nn.ModuleList([encoders[0]] * n_mols)
        else:
            self.encoders = nn.ModuleList(encoders)

    @property
    def output_dim(self) -> int:
        return sum(encoder.output_dim for encoder in self.encoders)

    def forward(self, reactant_batches: Iterable[MoleculeEncoderInput]) -> Tensor:
        """Encode the reactant_batch

        Parameters
        ----------
        reactant_batch : Iterable[MoleculeEncoderInput]
            an Iterable of length `n` containing inputs to a MoleculeEncoder, where `n` is the
            number of molecules in each reaction (== `self.n_mols`). I.e., to encode a batch of
            3-component reactions, the 0th entry of `reactant_batches` will be the batched inputs of
            the 0th component of each reaction, the 1st entry will be the 1st component, etc. In
            other words, if you were to (hypothetically) encode a batch of single component
            reactions (i.e., a batch of molecules), the only difference between a ReactionEncoder
            and a MoleculeEncoder would be the call signature:

            >>> inputs: tuple = X_v, X_e, a2b, ..., X_vd
            >>> mol_enc: MoleculeEncoder
            >>> rxn_enc: ReactionEncoder
            >>> H_mol = mol_enc(*inputs)
            >>> H_rxn = rxn_enc([inputs])
            >>> H_mol.shape == H_rxn.shape
            True

        Returns
        -------
        Tensor
            a Tensor of shape `b x d_o` containing the reaction encodings, where `b` is the number
            of reactions in the batch, and `d_o` is the `output_dim` of this encoder
            (== `self.n_mols x self.encoders[0].output_dim`)
        """
        Hs = [encoder(*inputs) for encoder, inputs in zip(self.encoders, reactant_batches)]
        H = torch.cat(Hs, 1)

        return H

    @classmethod
    def from_mol_encoder_args(
        cls, n_mols: int, shared: bool = False, *args, **kwargs
    ) -> ReactionEncoder:
        if not shared:
            encoders = [molecule_encoder(*args, **kwargs) for _ in range(n_mols)]
        else:
            encoders = [molecule_encoder(*args, **kwargs)]

        return cls(encoders, n_mols, shared)


def reaction_encoder(
    n_mols: int, shared: bool = False, *args, **kwargs
):
    if not shared:
        encoders = [molecule_encoder(*args, **kwargs) for _ in range(n_mols)]
    else:
        encoders = [molecule_encoder(*args, **kwargs)]

    return ReactionEncoder(encoders, n_mols, shared)