from abc import ABC, abstractmethod

from torch import nn


class MessagePassingBlock(nn.Module, ABC):
    def __len__(self) -> int:
        """an alias for the output dimension of the encoder"""
        return self.output_dim

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass
