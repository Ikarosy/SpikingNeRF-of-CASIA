import torch
import numpy as np
def minmaxNorm(input_tensor):
    return (input_tensor - torch.min(input_tensor)) / (torch.max(input_tensor) - torch.min(input_tensor))

def poisson_encoding(input_tensor):
    input_tensor = minmaxNorm(input_tensor)
    return (input_tensor > torch.rand(input_tensor.size(), device=input_tensor.device)).float()