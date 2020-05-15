from abc import ABC, abstractclassmethod
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
    Categorical,
    MixtureSameFamily,
    Independent,
    LowRankMultivariateNormal,
    MultivariateNormal,
    TransformedDistribution,
    AffineTransform,
)

from pts.core.component import validated
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

    @abstractclassmethod
    def domain_map(cls, *args: torch.Tensor):
        pass


class DistributionOutput(Output, ABC):

    distr_cls: type

    @validated()
    def __init__(self) -> None:
        pass

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:

        distr = self.distr_cls(*distr_args)
        if scale is None:
            return distr
        else:
            return TransformedDistribution(distr, [AffineTransform(loc=0, scale=scale)])


class NormalOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    distr_cls: type = Normal

    @classmethod
    def domain_map(cls, loc, scale):
        scale = F.softplus(scale)
        return loc.squeeze(-1), scale.squeeze(-1)

    @property
    def event_shape(self) -> Tuple:
        return ()


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
            # alpha = alpha + (scale - 1) / (scale * mu) # multiply 2nd moment by scale
            alpha += (scale - 1) / mu

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


class StudentTMixtureOutput(DistributionOutput):
    def __init__(self, components: int = 1) -> None:
        self.components = components
        self.args_dim = {
            "mix_logits": components,
            "df": components,
            "loc": components,
            "scale": components,
        }

    @classmethod
    def domain_map(cls, mix_logits, df, loc, scale):
        scale = F.softplus(scale)
        df = 2.0 + F.softplus(df)
        return (
            mix_logits.squeeze(-1),
            df.squeeze(-1),
            loc.squeeze(-1),
            scale.squeeze(-1),
        )

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        mix_logits, df, loc, scale = distr_args

        distr = MixtureSameFamily(
            Categorical(logits=mix_logits), StudentT(df, loc, scale)
        )
        if scale is None:
            return distr
        else:
            return TransformedDistribution(distr, [AffineTransform(loc=0, scale=scale)])

    @property
    def event_shape(self) -> Tuple:
        return ()


class NormalMixtureOutput(DistributionOutput):
    def __init__(self, components: int = 1) -> None:
        self.components = components
        self.args_dim = {
            "mix_logits": components,
            "loc": components,
            "scale": components,
        }

    @classmethod
    def domain_map(cls, mix_logits, loc, scale):
        scale = F.softplus(scale)
        return mix_logits.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    def distribution(
        self, distr_args, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        mix_logits, loc, scale = distr_args

        distr = MixtureSameFamily(Categorical(logits=mix_logits), Normal(loc, scale))
        if scale is None:
            return distr
        else:
            return TransformedDistribution(distr, [AffineTransform(loc=0, scale=scale)])

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

    @classmethod
    def domain_map(cls, loc, cov_factor, cov_diag):
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

    @classmethod
    def domain_map(cls, loc, scale):
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

    @classmethod
    def domain_map(cls, loc, scale):
        d = len(loc)
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


class FlowOutput(DistributionOutput):
    def __init__(self, flow, input_size, cond_size):
        self.args_dim = {"cond": cond_size}
        self.flow = flow
        self.dim = input_size

    @classmethod
    def domain_map(cls, cond):
        return (cond,)

    def distribution(self, distr_args, scale=None):
        (cond,) = distr_args
        if scale is not None:
            self.flow.scale = scale
        self.flow.cond = cond

        return self.flow

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
