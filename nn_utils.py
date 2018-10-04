from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


def create_mask(lengths: List[int], cuda: bool = False) -> torch.Tensor:
    """
    Creates a mask from a list of sequence lengths to mask out padding with 1s for content and 0s for padding.

    Example:
        >>> lengths = [3, 4, 2]
        >>> create_mask(lengths)
        tensor([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
                [0, 1, 0]], dtype=torch.uint8)

    :param lengths: A List of length `batch_size` with the length of each sequence in the batch.
    :param cuda: A boolean indicating whether to move the mask to GPU.
    :return: A tensor of shape `(sequence_length, batch_size)` with 1s for content and 0s for padding.
    """
    # Get sizes
    seq_len, batch_size = max(lengths), len(lengths)

    # Create length and index masks
    lengths = torch.LongTensor(lengths)
    length_mask = lengths.unsqueeze(0).repeat(seq_len, 1)  # (seq_len, batch_size)
    index_mask = torch.arange(seq_len, dtype=torch.long).unsqueeze(1).repeat(1, batch_size)  # (seq_len, batch_size)

    # Create mask
    mask = (index_mask < length_mask).float()

    # Move to GPU
    if cuda:
        mask = mask.cuda()

    return mask


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: int,
                 total_epochs: int,
                 steps_per_epoch: int,
                 init_lr: float,
                 max_lr: float,
                 final_lr: float):
        """
        Initializes the learning rate scheduler.

        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return [self.lr]

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        if self.current_step <= self.warmup_steps:
            self.lr = self.init_lr + self.current_step * self.linear_increment
        else:
            self.lr = self.max_lr * (self.exponential_gamma ** (self.current_step - self.warmup_steps))

        self.optimizer.param_groups[0]['lr'] = self.lr


def GRU(x: torch.Tensor, h_nei: torch.Tensor, W_z: nn.Linear, W_r: nn.Linear, U_r: nn.Linear, W_h: nn.Linear) -> torch.Tensor:
    hidden_size = x.size()[-1]
    sum_h = h_nei.sum(dim=1)
    z_input = torch.cat([x, sum_h], dim=1)
    z = nn.Sigmoid()(W_z(z_input))

    r_1 = W_r(x).view(-1, 1, hidden_size)
    r_2 = U_r(h_nei)
    r = nn.Sigmoid()(r_1 + r_2)

    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(dim=1)
    h_input = torch.cat([x, sum_gated_h], dim=1)
    pre_h = nn.Tanh()(W_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h

    return new_h


class GraphGRU(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, depth: int):
        super(GraphGRU, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h: torch.Tensor, x: torch.Tensor, mess_graph: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(h.size(0), 1)
        mask[0] = 0  # first vector is padding
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        for it in range(self.depth):
            h_nei = index_select_ND(h, mess_graph)
            sum_h = h_nei.sum(dim=1)
            z_input = torch.cat([x, sum_h], dim=1)
            z = F.sigmoid(self.W_z(z_input))

            r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
            r_2 = self.U_r(h_nei)
            r = F.sigmoid(r_1 + r_2)

            gated_h = r * h_nei
            sum_gated_h = gated_h.sum(dim=1)
            h_input = torch.cat([x, sum_gated_h], dim=1)
            pre_h = F.tanh(self.W_h(h_input))
            h = (1.0 - z) * sum_h + z * pre_h
            h = h * mask

        return h
