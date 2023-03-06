import warnings
from typing import Dict, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions import (
    Beta,
    Categorical,
    Distribution,
    Independent,
    LowRankMultivariateNormal,
    MixtureSameFamily,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    Poisson,
)

from gluonts.core.component import validated
from gluonts.torch.distributions import AffineTransformed, DistributionOutput
from gluonts.torch.distributions.studentT import StudentT

from pts.distributions import ZeroInflatedNegativeBinomial, ZeroInflatedPoisson


class IndependentDistributionOutput(DistributionOutput):
    @validated()
    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim

    @property
    def event_shape(self) -> Tuple:
        if self.dim is None:
            return ()
        else:
            return (self.dim,)

    def independent(self, distr: Distribution) -> Distribution:
        if self.dim is None:
            return distr

        return Independent(distr, 1)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        distr = self.independent(self.distr_cls(*distr_args))
        if scale is None:
            scale = 1.0
        if loc is None:
            loc = 0.0
        return AffineTransformed(distr, loc=loc, scale=scale)

    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        return (x + torch.sqrt(torch.square(x) + 4.0)) / 2.0


class NormalOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    distr_cls: type = Normal

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, loc, scale):
        scale = cls.squareplus(scale)
        return loc.squeeze(-1), scale.squeeze(-1)


class IndependentNormalOutput(NormalOutput):
    @validated()
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        warnings.warn(
            "IndependentNormalOutput is deprecated. Use NormalOutput instead.",
            DeprecationWarning,
        )


class BetaOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"concentration1": 1, "concentration0": 1}
    distr_cls: type = Beta

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, concentration1, concentration0):
        concentration1 = cls.squareplus(concentration1) + 1e-8
        concentration0 = cls.squareplus(concentration0) + 1e-8
        return concentration1.squeeze(-1), concentration0.squeeze(-1)


class PoissonOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"rate": 1}
    distr_cls: type = Poisson

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, rate):
        rate_pos = cls.squareplus(rate).clone()

        return (rate_pos.squeeze(-1),)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        (rate,) = distr_args

        if scale is not None:
            rate *= scale

        if loc is not None:
            rate += loc

        return self.independent(Poisson(rate))


class ZeroInflatedPoissonOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"gate": 1, "rate": 1}
    distr_cls: type = ZeroInflatedPoisson

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, gate, rate):
        gate_unit = torch.sigmoid(gate).clone()
        rate_pos = cls.squareplus(rate).clone()

        return gate_unit.squeeze(-1), rate_pos.squeeze(-1)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        gate, rate = distr_args

        if scale is not None:
            rate *= scale

        if loc is not None:
            rate += loc

        return self.independent(ZeroInflatedPoisson(gate=gate, rate=rate))


class NegativeBinomialOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"total_count": 1, "logits": 1}
    distr_cls: type = NegativeBinomial

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, total_count, logits):
        total_count = cls.squareplus(total_count)
        return total_count.squeeze(-1), logits.squeeze(-1)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        total_count, logits = distr_args

        if scale is not None:
            logits += scale.log()

        # shift negative binomial parameters by loc
        if loc is not None:
            total_count += loc

        return self.independent(
            NegativeBinomial(total_count=total_count, logits=logits)
        )


class ZeroInflatedNegativeBinomialOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"gate": 1, "total_count": 1, "logits": 1}
    distr_cls: type = ZeroInflatedNegativeBinomial

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, gate, total_count, logits):
        gate = torch.sigmoid(gate)
        total_count = cls.squareplus(total_count)
        return gate.squeeze(-1), total_count.squeeze(-1), logits.squeeze(-1)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        gate, total_count, logits = distr_args

        if scale is not None:
            logits += scale.log()

        # shift negative binomial parameters by loc
        if loc is not None:
            total_count += loc

        return self.independent(
            ZeroInflatedNegativeBinomial(
                gate=gate, total_count=total_count, logits=logits
            )
        )


class StudentTOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    distr_cls: type = StudentT

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        if dim is not None:
            self.args_dim = {k: dim for k in self.args_dim}

    @classmethod
    def domain_map(cls, df, loc, scale):
        scale = cls.squareplus(scale)
        df = 2.0 + cls.squareplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)


class StudentTMixtureOutput(DistributionOutput):
    @validated()
    def __init__(self, components: int = 1) -> None:
        self.components = components
        self.args_dim = {
            "mix_logits": components,
            "df": components,
            "loc": components,
            "scale": components,
        }

    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        return (x + torch.sqrt(torch.square(x) + 4.0)) / 2.0

    @classmethod
    def domain_map(cls, mix_logits, df, loc, scale):
        scale = cls.squareplus(scale)
        df = 2.0 + cls.squareplus(df)
        return (
            mix_logits.squeeze(-1),
            df.squeeze(-1),
            loc.squeeze(-1),
            scale.squeeze(-1),
        )

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        mix_logits, df, dist_loc, dist_scale = distr_args

        distr = MixtureSameFamily(
            Categorical(logits=mix_logits), StudentT(df, dist_loc, dist_scale)
        )
        if scale is None:
            scale = 1.0
        if loc is None:
            loc = 0.0
        return AffineTransformed(distr, loc=loc, scale=scale)

    @property
    def event_shape(self) -> Tuple:
        return ()


class NormalMixtureOutput(DistributionOutput):
    @validated()
    def __init__(self, components: int = 1) -> None:
        self.components = components
        self.args_dim = {
            "mix_logits": components,
            "loc": components,
            "scale": components,
        }

    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        return (x + torch.sqrt(torch.square(x) + 4.0)) / 2.0

    @classmethod
    def domain_map(cls, mix_logits, loc, scale):
        scale = cls.squareplus(scale)
        return mix_logits.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        mix_logits, dist_loc, dist_scale = distr_args

        distr = MixtureSameFamily(
            Categorical(logits=mix_logits), Normal(dist_loc, dist_scale)
        )
        if scale is None:
            scale = 1.0
        if loc is None:
            loc = 0.0
        return AffineTransformed(distr, loc=loc, scale=scale)

    @property
    def event_shape(self) -> Tuple:
        return ()


class LowRankMultivariateNormalOutput(DistributionOutput):
    @validated()
    def __init__(
        self,
        dim: int,
        rank: int,
        sigma_init: float = 1.0,
        sigma_minimum: float = 1e-3,
    ) -> None:
        self.distr_cls = LowRankMultivariateNormal
        self.dim = dim
        self.rank = rank
        self.sigma_init = sigma_init
        self.sigma_minimum = sigma_minimum
        self.args_dim = {"loc": dim, "cov_factor": dim * rank, "cov_diag": dim}

    def domain_map(self, loc, cov_factor, cov_diag):
        diag_bias = (
            self.inv_softplus(self.sigma_init**2) if self.sigma_init > 0.0 else 0.0
        )

        shape = cov_factor.shape[:-1] + (self.dim, self.rank)
        cov_factor = cov_factor.reshape(shape)
        cov_diag = self.squareplus(cov_diag + diag_bias) + self.sigma_minimum**2

        return loc, cov_factor, cov_diag

    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        return (x + torch.sqrt(torch.square(x) + 4.0)) / 2.0

    @staticmethod
    def inv_softplus(y):
        if y < 20.0:
            return np.log(np.exp(y) - 1.0)
        else:
            return y

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)


class MultivariateNormalOutput(DistributionOutput):
    @validated()
    def __init__(self, dim: int) -> None:
        self.args_dim = {"loc": dim, "scale_tril": dim * dim}
        self.dim = dim

    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        return (x + torch.sqrt(torch.square(x) + 4.0)) / 2.0

    def domain_map(self, loc, scale):
        d = self.dim
        device = scale.device

        shape = scale.shape[:-1] + (d, d)
        scale = scale.reshape(shape)

        scale_diag = self.squareplus(scale * torch.eye(d, device=device)) * torch.eye(
            d, device=device
        )

        mask = torch.tril(torch.ones_like(scale), diagonal=-1)
        scale_tril = (scale * mask) + scale_diag

        return loc, scale_tril

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        dist_loc, scale_tri = distr_args
        distr = MultivariateNormal(loc=dist_loc, scale_tril=scale_tri)

        if scale is None:
            scale = 1.0
        if loc is None:
            loc = 0.0
        return AffineTransformed(distr, loc=loc, scale=scale)

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)


class FlowOutput(DistributionOutput):
    @validated()
    def __init__(self, flow, input_size, cond_size):
        self.args_dim = {"cond": cond_size}
        self.flow = flow
        self.dim = input_size

    @classmethod
    def domain_map(cls, cond):
        return (cond,)

    def distribution(self, distr_args, loc=None, scale=None):
        (cond,) = distr_args
        if scale is not None:
            self.flow.scale = scale
        if loc is not None:
            self.flow.loc = loc
        self.flow.cond = cond

        return self.flow

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)


class DiffusionOutput(DistributionOutput):
    @validated()
    def __init__(self, diffusion, input_size, cond_size):
        self.args_dim = {"cond": cond_size}
        self.diffusion = diffusion
        self.dim = input_size

    @classmethod
    def domain_map(cls, cond):
        return (cond,)

    def distribution(self, distr_args, loc=None, scale=None):
        (cond,) = distr_args
        if scale is not None:
            self.diffusion.scale = scale
        if loc is not None:
            self.diffusion.loc = loc
        self.diffusion.cond = cond

        return self.diffusion

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
