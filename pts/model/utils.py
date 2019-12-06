import inspect

import torch.nn as nn


def get_module_forward_input_names(module: nn.Module):
    params = inspect.signature(module.forward).parameters
    return list(params)


def copy_parameters(net_source: nn.Module, net_dest: nn.Module) -> None:
    net_dest.load_state_dict(net_source.state_dict())
