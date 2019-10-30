from typing import Callable, Dict, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn


class ArgProj(nn.Module):
    def __init__(
        self,
        in_features,
        args_dim: Dict[str, int],
        domain_map: Callable[..., Tuple[torch.Tensor]],
        dtype: np.dtype = np.float32,
        prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.dtype = dtype
        self.proj = nn.ModuleList(
            [nn.Linear(in_features, dim) for dim in args_dim.values()]
        )
        self.domain_map = domain_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]

        return self.domain_map(*params_unbounded)

class Output(ABC):
    