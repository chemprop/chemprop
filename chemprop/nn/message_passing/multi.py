import logging
from typing import Iterable, Sequence

from torch import Tensor, nn

from chemprop.data import BatchMolGraph
from chemprop.nn.hparams import HasHParams
from chemprop.nn.message_passing.proto import MessagePassing

logger = logging.getLogger(__name__)


class MulticomponentMessagePassing(nn.Module, HasHParams):
    """A `MulticomponentMessagePassing` performs message-passing on each individual input in a
    multicomponent input then concatenates the representation of each input to construct a
    global representation

    Parameters
    ----------
    blocks : Sequence[MessagePassing]
        the invidual message-passing blocks for each input
    n_components : int
        the number of components in each input
    shared : bool, default=False
        whether one block will be shared among all components in an input. If not, a separate
        block will be learned for each component.
    """

    def __init__(self, blocks: Sequence[MessagePassing], n_components: int, shared: bool = False):
        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "blocks": [block.hparams for block in blocks],
            "n_components": n_components,
            "shared": shared,
        }

        if len(blocks) == 0:
            raise ValueError("arg 'blocks' was empty!")
        if shared and len(blocks) > 1:
            logger.warning(
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

        return d_o

    def forward(self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor | None]) -> list[Tensor]:
        """Encode the multicomponent inputs

        Parameters
        ----------
        bmgs : Iterable[BatchMolGraph]
        V_ds : Iterable[Tensor | None]

        Returns
        -------
        list[Tensor]
            a list of tensors of shape `V x d_i` containing the respective encodings of the `i`\th
            component, where `d_i` is the output dimension of the `i`\th encoder
        """
        if V_ds is None:
            return [block(bmg) for block, bmg in zip(self.blocks, bmgs)]
        else:
            return [block(bmg, V_d) for block, bmg, V_d in zip(self.blocks, bmgs, V_ds)]
