from __future__ import annotations

import warnings
from typing import Iterable

import torch
from torch import Tensor, nn

from chemprop.v2.models.modules.message_passing.base import MessagePassingBlock
from chemprop.v2.models.modules.message_passing.molecule import (
    MolecularMessagePassingBlock,
    MolecularInput,
    molecule_block,
)

ReactionInput = Iterable[MolecularInput]


class CompositeMessagePassingBlock(MessagePassingBlock):
    """A `CompositeMessagePassingBlock` performs message-passing on each individual input in a
    multicomponent input then concatenates the representation of each input to construct a
    global representation

    Inputs
    ------
    blocks : Iterable[MolecularMessagePassingBlock]
        the invidual message-passing blocks for each input
    n_components : int
        the number of components in each input
    shared : bool, default=False
        whether one block will be shared among all components in an input. If not, a separate
        block will be learned for each component.
    """

    def __init__(
        self,
        blocks: Iterable[MolecularMessagePassingBlock],
        n_components: int,
        shared: bool = False,
    ):
        super().__init__()

        if len(blocks) == 0:
            raise ValueError("arg 'blocks' was empty!")

        if shared and len(blocks) > 1:
            warnings.warn(
                "More than 1 block was supplied but 'shared' was True! Using only the 0th block..."
            )
        elif not shared and len(blocks) != n_components:
            raise ValueError(
                "arg 'n_components' must be equal to `len(blocks)` if 'shared' is False! "
                f"got: {n_components} and {len(blocks)}, respectively."
            )

        self.n_components = n_components
        self.shared = shared

        if self.shared:
            self.blocks = nn.ModuleList([blocks[0]] * self.n_components)
        else:
            self.blocks = nn.ModuleList(blocks)

    @property
    def output_dim(self) -> int:
        return sum(block.output_dim for block in self.blocks)

    def forward(self, inputss: Iterable[MolecularInput]) -> Tensor:
        """Encode the multicomponent inputs

        Parameters
        ----------
        inputss : Iterable[MolecularInput]
            an Iterable of length `n` containing inputs to a `MolecularMessagePassingBlock`, where
            `n` is the number of components in each reaction (== `self.n_components`). I.e., to
            encode a batch of 3-component inputs, the 0th entry of `inputss` will be the batched
            inputs of the 0th component of each input, the 1st entry will be the 1st component,
            etc. Each component would itself be a batch of length `b`. In other words, if you were
            to (hypothetically) encode a batch of single component inputs (i.e., a batch of
            molecules), the only difference between a `CompositeMessagePassingBlock` and a
            `MolecularMessagePassingBlock` would be the call signature

        Example
        -------
        >>> inputs: tuple[BatchMolGraph, Optional[Tensor]]
        >>> block = MolecularMessagePassingBlock()
        >>> multi_block = CompositeMessagePassingBlock([block], n_mols=1)
        >>> H_single = block(*inputs)
        >>> H_multi = multi_block([inputs])
        >>> H_single.shape == H_multi.shape
        True

        Returns
        -------
        Tensor
            a Tensor of shape `b x d_o` containing the reaction encodings, where `b` is the number
            of reactions in the batch, and `d_o` is the `output_dim` of this encoder
            (== `self.n_mols x self.encoders[0].output_dim`)
        """
        Hs = [block(*inputs) for block, inputs in zip(self.blocks, inputss)]
        H = torch.cat(Hs, 1)

        return H


def composite_block(n_components: int, shared: bool = False, *args, **kwargs):
    if not shared:
        encoders = [molecule_block(*args, **kwargs) for _ in range(n_components)]
    else:
        encoders = [molecule_block(*args, **kwargs)]

    return CompositeMessagePassingBlock(encoders, n_components, shared)
