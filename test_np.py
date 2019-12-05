import torch
import torch.nn as nn
import numpy as np

# W_att = nn.Linear(4, 1)
arr = torch.ones((2,3))
arr = arr.unsqueeze(dim=2).repeat(1,1,4)
arr = arr.sum(dim=2)
print(arr)
# ans = W_att(arr)
# print(W_att, arr, ans)