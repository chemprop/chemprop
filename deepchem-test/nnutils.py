import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.
    
    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor).cuda()
    else:
        return Variable(tensor, requires_grad=requires_grad).cuda()


def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)


def GRU(x, h_nei, W_z, W_r, U_r, W_h):
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

    def __init__(self, input_size, hidden_size, depth):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h, x, mess_graph):
        mask = torch.ones(h.size(0), 1)
        mask[0] = 0  # first vector is padding
        mask = create_var(mask)
        for it in xrange(self.depth):
            h_nei = index_select_ND(h, 0, mess_graph)
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
