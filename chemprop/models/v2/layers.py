from torch import Tensor, nn

class Exp(nn.Module):
    def forward(self, x: Tensor):
        return x.exp()