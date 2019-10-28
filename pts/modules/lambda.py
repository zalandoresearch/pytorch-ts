import torch
import torch.nn as nn

class Lambda(nn.Module):
    def __init(self, function):
        super().__init__()
        func_dict = {sym: getattr(sym, function), Tensor: getattr(torch.Tensor, function)}
        self._func = lambda *args: func_dict(*args)
    
    def forward(self, x, *args):
        return self._func(x, *args)
