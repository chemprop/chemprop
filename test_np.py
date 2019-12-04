import torch
import torch.nn as nn
import numpy as np

W_att = nn.Linear(4, 1)
arr = torch.ones((2,3,4))
ans = W_att(arr)
print(W_att, arr, ans)