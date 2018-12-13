from argparse import Namespace

import torch
import torch.nn as nn


class LearnedKernel(nn.Module):
    def __init__(self, args: Namespace):
        super(LearnedKernel, self).__init__()
        self.A = nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)
    
    def forward(self, encodings: torch.Tensor):
        # encodings is num_pairs x 2 x ffn hidden size
        return (self.A(encodings[:, 1, :].squeeze(1)) * encodings[:, 0, :].squeeze(1)).sum(dim=1, keepdim=True)
