import torch
import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self, function):
        super().__init__()
        self._func = function
    
    def forward(self, x, *args):
        return self._func(x, *args)
