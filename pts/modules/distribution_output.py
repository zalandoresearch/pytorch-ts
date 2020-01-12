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
    Normal,
    Independent,
    LowRankMultivariateNormal,
    MultivariateNormal,
    TransformedDistribution,
    AffineTransform,
)

from .lambda_layer import LambdaLayer
from .flows import RealNVP


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


class DistributionOutput(Output, ABC):
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


class LowRankMultivariateNormalOutput(DistributionOutput):
    def __init__(
        self, dim: int, rank: int, sigma_init: float = 1.0, sigma_minimum: float = 1e-3,
    ) -> None:
        self.distr_cls = LowRankMultivariateNormal
        self.dim = dim
        self.rank = rank
        self.sigma_init = sigma_init
        self.sigma_minimum = sigma_minimum
        self.args_dim = {"loc": dim, "cov_factor": dim * rank, "cov_diag": dim}

    def domain_map(self, loc, cov_factor, cov_diag):
        diag_bias = (
            self.inv_softplus(self.sigma_init ** 2) if self.sigma_init > 0.0 else 0.0
        )

        shape = cov_factor.shape[:-1] + (self.dim, self.rank)
        cov_factor = cov_factor.reshape(shape)
        cov_diag = F.softplus(cov_diag + diag_bias) + self.sigma_minimum ** 2

        return loc, cov_factor, cov_diag

    def inv_softplus(self, y):
        if y < 20.0:
            return np.log(np.exp(y) - 1.0)
        else:
            return y

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)


class IndependentNormalOutput(DistributionOutput):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.args_dim = {"loc": self.dim, "scale": self.dim}

    def domain_map(self, loc, scale):
        return loc, F.softplus(scale)

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        distr = Independent(Normal(*distr_args), 1)

        if scale is None:
            return distr
        else:
            return TransformedDistribution(distr, [AffineTransform(loc=0, scale=scale)])


class MultivariateNormalOutput(DistributionOutput):
    def __init__(self, dim: int) -> None:
        self.args_dim = {"loc": dim, "scale_tril": dim * dim}
        self.dim = dim

    def domain_map(self, loc, scale):
        d = self.dim
        device = scale.device

        shape = scale.shape[:-1] + (d, d)
        scale = scale.reshape(shape)

        scale_diag = F.softplus(scale * torch.eye(d, device=device)) * torch.eye(
            d, device=device
        )

        mask = torch.tril(torch.ones_like(scale), diagonal=-1)
        scale_tril = (scale * mask) + scale_diag

        return loc, scale_tril

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        loc, scale_tri = distr_args
        distr = MultivariateNormal(loc=loc, scale_tril=scale_tri)

        if scale is None:
            return distr
        else:
            return TransformedDistribution(distr, [AffineTransform(loc=0, scale=scale)])


    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)



class RealNVPOutput(DistributionOutput):
    def __init__(self, input_size, cond_size, n_blocks=3, n_hidden=1, hidden_size=100):
        self.args_dim = {"cond": cond_size}
        self.dim = input_size
        self.flow = RealNVP(
            n_blocks=n_blocks,
            input_size=input_size,
            hidden_size=hidden_size,
            n_hidden=n_hidden,
            cond_label_size=cond_size 
        )
    
    def domain_map(self, cond):
        return (cond,)

    def distribution(self, distr_args, scale=None):
        cond, = distr_args
        self.flow.cond = cond
    
        return self.flow

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
