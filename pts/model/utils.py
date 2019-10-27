import inspect

import torch.nn as nn


def get_module_forward_input_names(module: nn.Module):
    params = inspect.signature(module.forward).parameters
    return list(params)
