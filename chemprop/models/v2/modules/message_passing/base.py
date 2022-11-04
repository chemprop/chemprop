from abc import ABC, abstractmethod

from torch import nn


class MessagePassingBlock(ABC, nn.Module):
    def __len__(self) -> int:
        """the output dimension of the message passing block"""
        return self.output_dim

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass
