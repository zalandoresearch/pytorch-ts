from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution, TransformedDistribution, AffineTransform

from .lambda_layer import LambdaLayer


class ArgProj(nn.Module):
    def __init__(
        self,
        in_features: int,
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
    in_features: int
    args_dim: Dict[str, int]
    _dtype: np.dtype = np.float32

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: np.dtype):
        self._dtype = dtype

    def get_args_proj(self, prefix: Optional[str] = None) -> ArgProj:
        return ArgProj(
            in_features=self.in_features,
            args_dim=self.args_dim,
            domain_map=LambdaLayer(self.domain_map),
            prefix=prefix,
            dtype=self.dtype,
        )

    @abstractmethod
    def domain_map(self, *args: torch.Tensor):
        pass


class DistributionOutput(Output):

    distr_cls: type

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:

        if scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return TransformedDistribution(distr, [AffineTransform(loc=0, scale=scale)])
