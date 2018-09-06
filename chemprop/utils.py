import torch

def cuda_available():
    try:
        has_cuda = torch.cuda.get_device_capability(torch.cuda.current_device())[0] > 3
    except AssertionError as e:
        has_cuda = False
    return has_cuda