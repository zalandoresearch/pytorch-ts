from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Beta,
    NegativeBinomial,
    StudentT,
    TransformedDistribution,
    AffineTransform,
)

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

    def get_args_proj(self, in_features: int, prefix: Optional[str] = None) -> ArgProj:
        return ArgProj(
            in_features=in_features,
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


class BetaOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"concentration1": 1, "concentration0": 1}
    distr_cls: type = Beta

    @classmethod
    def domain_map(cls, concentration1, concentration0):
        concentration1 = F.softplus(concentration1) + 1e-8
        concentration0 = F.softplus(concentration0) + 1e-8
        return concentration1.squeeze(-1), concentration0.squeeze(-1)
    
    @property
    def event_shape(self) -> Tuple:
        return ()


class NegativeBinomialOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"mu": 1, "alpha": 1}
    distr_cls: Distribution = NegativeBinomial

    @classmethod
    def domain_map(cls, mu, alpha):
        mu = F.softplus(mu) + 1e-8
        alpha = F.softplus(alpha) + 1e-8
        return mu.squeeze(-1), alpha.squeeze(-1)

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        mu, alpha = distr_args

        if scale is not None:
            mu *= scale
            alpha *= torch.sqrt(scale + 1.0)

        n = 1.0 / alpha
        p = mu * alpha / (1.0 + mu * alpha)

        return NegativeBinomial(total_count=n, probs=p)
    
    @property
    def event_shape(self) -> Tuple:
        return ()


class StudentTOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    distr_cls: type = StudentT

    @classmethod
    def domain_map(cls, df, loc, scale):
        scale = F.softplus(scale)
        df = 2.0 + F.softplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
