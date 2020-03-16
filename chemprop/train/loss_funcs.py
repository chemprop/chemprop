"""Custom loss functions."""

import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):

    def __init__(self):
        """
        Initializes contrastive loss with no reduction.
        """
        super(ContrastiveLoss, self).__init__()
        self.logsoftmax=nn.LogSoftmax(dim=0)

    def get_lengths(self, targets):
        ret, size = [], 0
        for target in targets.flatten().tolist():
            if target:
                ret.append(size)
                size = 1
            else:
                size += 1
        ret.append(size)
        return ret[1:]  # Adds dummy size in the very beginning

    def forward(self, preds, targets):
        """
        Returns contrastive loss. Assumes that batches are marked by one positive pair.

        :param preds: Scoring output by the model.
        :param targets: Truth for cmpds.
        :return: Contrastive loss for scores.
        """
        lengths = self.get_lengths(targets)
        ret, start = [], 0

        for size in lengths:
            ret.append(self.logsoftmax( preds[start:start+size,:] ))
            start += size
        return -torch.cat(ret)
