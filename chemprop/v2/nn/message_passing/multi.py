from typing import Iterable, Sequence
import warnings

from torch import Tensor, nn

from chemprop.v2.data import BatchMolGraph
from chemprop.v2.nn.message_passing.base import MessagePassingBlock


class MulticomponentMessagePassing(nn.Module):
    """A `MulticomponentMessagePassing` performs message-passing on each individual input in a
    multicomponent input then concatenates the representation of each input to construct a
    global representation

    Parameters
    ----------
    blocks : Sequence[MolecularMessagePassingBlock]
        the invidual message-passing blocks for each input
    n_components : int
        the number of components in each input
    shared : bool, default=False
        whether one block will be shared among all components in an input. If not, a separate
        block will be learned for each component.
    """

    def __init__(
        self, blocks: Sequence[MessagePassingBlock], n_components: int, shared: bool = False
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
        self.blocks = nn.ModuleList([blocks[0]] * self.n_components if shared else blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def output_dim(self) -> int:
        d_o = sum(block.output_dim for block in self.blocks)

        return d_o if not self.shared else self.blocks[0].output_dim

    def forward(self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor | None]) -> list[Tensor]:
        """Encode the multicomponent inputs

        Parameters
        ----------
        bmgs : Iterable[BatchMolGraph]
        V_ds : Iterable[Tensor | None]

        Returns
        -------
        list[Tensor]
            a list of tensors of shape `b x d_i` containing the respective encodings of the `i`th component, where `b` is the number of components in the batch, and `d_i` is the output dimension of the `i`th encoder
        """
        return [block(bmg, V_d) for block, bmg, V_d in zip(self.blocks, bmgs, V_ds)]
