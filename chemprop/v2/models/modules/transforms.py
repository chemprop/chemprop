from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch.nn import functional as F

class OutputTransform(nn.Module, ABC):
    n_targets: int

    @abstractmethod
    def forward(self, Y: Tensor) -> Tensor:
        pass


class RegressionTransform(OutputTransform):
    n_targets = 1

    def __init__(self, loc: float | Tensor = 0, scale: float | Tensor = 1) -> None:
        super().__init__()

        self.loc: Tensor = torch.atleast_2d(torch.tensor(loc))
        self.scale: Tensor = torch.atleast_2d(torch.tensor(scale))
    
    def forward(self, Y: Tensor) -> Tensor:
        if self.training:
            return Y
        
        return self.scale * Y + self.loc


class MVETransform(RegressionTransform):
    n_targets = 2

    def forward(self, Y: Tensor) -> Tensor:
        mean, var = torch.chunk(Y, self.n_targets, 1)

        if self.training:
            var = F.softplus(var)
        else:
            mean = self.scale * mean + self.loc
            var = var * self.scale**2

        return torch.cat((mean, var), 1)


class EvidentialTransform(RegressionTransform):
    n_targets = 4

    def forward(self, Y: Tensor) -> Tensor:
        mean, v, alpha, beta = torch.chunk(Y, self.n_targets, 1)

        if self.training:
            v = F.softplus(v)
            alpha = F.softplus(alpha) + 1
            beta = F.softplus(beta)
        else:
            mean = self.scale * mean + self.loc
            v = v * self.scale**2

        return torch.cat((mean, v, alpha, beta), 1)


class BinaryTransform(OutputTransform):
    n_targets = 1

    def forward(self, Y: Tensor) -> Tensor:
        if self.training:
            return Y
        
        return Y.sigmoid()


class DirichletBinaryTransform(BinaryTransform):
    n_targets = 2

    def forward(self, Y: Tensor) -> Tensor:
        if self.training:
            return F.softplus(Y) + 1
        
        alpha, beta = torch.chunk(Y, 2, 1)
        
        return beta / (alpha + beta)
        

class MulticlassTransform(OutputTransform):
    n_targets = 1

    def __init__(self, n_classes: int) -> None:
        super().__init__()

        self.n_classes = n_classes

    def forward(self, Y: Tensor) -> Tensor:
        Y = Y.reshape(Y.shape[0], -1, self.n_classes)

        return Y if self.training else Y.softmax(-1)


class DirichletMulticlassTransform(MulticlassTransform):
    def forward(self, Y: Tensor) -> Tensor:
        Y = Y.reshape(Y.shape[0], -1, self.n_classes)

        if self.training:
            return F.softplus(Y) + 1
        
        Y = Y.softmax(-1)
        Y = F.softplus(Y) + 1
        
        alpha = Y
        Y = Y / Y.sum(-1, keepdim=True)

        return torch.cat((Y, alpha), 1)