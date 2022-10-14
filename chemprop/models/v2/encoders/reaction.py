from __future__ import annotations

import warnings
from typing import Iterable

import torch
from torch import Tensor, nn

from chemprop.models.v2.encoders.base import MessagePassingBlock
from chemprop.models.v2.encoders.molecule import (
    MolecularMessagePassingBlock,
    MolecularInput,
    molecule_block,
)

ReactionInput = Iterable[MolecularInput]

class ReactionMessagePassingBlock(MessagePassingBlock):
    def __init__(
        self, encoders: Iterable[MolecularMessagePassingBlock], n_mols: int, shared: bool = False
    ):
        super().__init__()

        if len(encoders) == 0:
            raise ValueError("arg 'encoders' was empty!")

        if shared and len(encoders) > 1:
            warnings.warn(
                "More than 1 encoder was supplied but 'shared' was True! "
                "Using only the 0th encoder..."
            )
        elif not shared and len(encoders) != n_mols:
            raise ValueError(
                "arg 'n_mols' must be equal to `len(encoders)` if 'shared' is False! "
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

    def forward(self, reactant_inputs: Iterable[MolecularInput]) -> Tensor:
        """Encode the reactant_batch

        Parameters
        ----------
        reactant_inputs : Iterable[MoleculeEncoderInput]
            an Iterable of length `n` containing inputs to a MoleculeEncoder, where `n` is the
            number of molecules in each reaction (== `self.n_mols`). I.e., to encode a batch of
            3-component reactions, the 0th entry of `reactant_batches` will be the batched inputs of
            the 0th component of each reaction, the 1st entry will be the 1st component, etc. In
            other words, if you were to (hypothetically) encode a batch of single component
            reactions (i.e., a batch of molecules), the only difference between a ReactionEncoder
            and a MoleculeEncoder would be the call signature:

        Example
        -------
        >>> batch: tuple[BatchMolGraph, Optional[Tensor]]
        >>> mol_enc = molecule_encoder()
        >>> rxn_enc = ReactionEncoder([mol_enc], n_mols=1)
        >>> H_mol = mol_enc(*batch)
        >>> H_rxn = rxn_enc([batch])
        >>> H_mol.shape == H_rxn.shape
        True

        Returns
        -------
        Tensor
            a Tensor of shape `b x d_o` containing the reaction encodings, where `b` is the number
            of reactions in the batch, and `d_o` is the `output_dim` of this encoder
            (== `self.n_mols x self.encoders[0].output_dim`)
        """
        Hs = [encoder(*inputs) for encoder, inputs in zip(self.encoders, reactant_inputs)]
        H = torch.cat(Hs, 1)

        return H


def reaction_encoder(n_mols: int, shared: bool = False, *args, **kwargs):
    if not shared:
        encoders = [molecule_block(*args, **kwargs) for _ in range(n_mols)]
    else:
        encoders = [molecule_block(*args, **kwargs)]

    return ReactionMessagePassingBlock(encoders, n_mols, shared)
