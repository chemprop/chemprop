from argparse import Namespace

import torch
import torch.nn as nn

class LearnedKernel(nn.Module):
    def __init__(self, args: Namespace):
        super(LearnedKernel, self).__init__()
        self.A = nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)
    
    def forward(self, encoding1: torch.Tensor, encoding2: torch.Tensor):
        return (self.A(encoding2) * encoding1).sum()