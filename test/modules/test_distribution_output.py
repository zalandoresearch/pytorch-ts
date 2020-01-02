import pytest
from typing import Iterable, List, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
from torch.distributions import (
    StudentT,
    Beta,
    NegativeBinomial,
    LowRankMultivariateNormal,
)

from pts.modules import (
    DistributionOutput,
    StudentTOutput,
    BetaOutput,
    NegativeBinomialOutput,
    LowRankMultivariateNormalOutput,
)

NUM_SAMPLES = 2000
BATCH_SIZE = 32
TOL = 0.3
START_TOL_MULTIPLE = 1


def inv_softplus(y: np.ndarray) -> np.ndarray:
    return np.log(np.exp(y) - 1)


def maximum_likelihood_estimate_sgd(
    distr_output: DistributionOutput,
    samples: torch.Tensor,
    init_biases: List[np.ndarray] = None,
    num_epochs: int = 5,
    learning_rate: float = 1e-2,
):
    arg_proj = distr_output.get_args_proj(in_features=1)

    if init_biases is not None:
        for param, bias in zip(arg_proj.proj, init_biases):
            nn.init.constant_(param.bias, bias)

    dummy_data = torch.ones((len(samples), 1))

    dataset = TensorDataset(dummy_data, samples)
    train_data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = SGD(arg_proj.parameters(), lr=learning_rate)

    for e in range(num_epochs):
        cumulative_loss = 0
        num_batches = 0

        for i, (data, sample_label) in enumerate(train_data):
            optimizer.zero_grad()
            distr_args = arg_proj(data)
            distr = distr_output.distribution(distr_args)
            loss = -distr.log_prob(sample_label).mean()
            loss.backward()
            clip_grad_norm_(arg_proj.parameters(), 10.0)
            optimizer.step()

            num_batches += 1
            cumulative_loss += loss.item()
        print("Epoch %s, loss: %s" % (e, cumulative_loss / num_batches))

    if len(distr_args[0].shape) == 1:
        return [param.detach().numpy() for param in arg_proj(torch.ones((1, 1)))]

    return [param[0].detach().numpy() for param in arg_proj(torch.ones((1, 1)))]


@pytest.mark.parametrize("concentration1, concentration0", [(3.75, 1.25)])
def test_beta_likelihood(concentration1: float, concentration0: float) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """

    # generate samples
    concentration1s = torch.zeros((NUM_SAMPLES,)) + concentration1
    concentration0s = torch.zeros((NUM_SAMPLES,)) + concentration0

    distr = Beta(concentration1s, concentration0s)
    samples = distr.sample()

    init_biases = [
        inv_softplus(concentration1 - START_TOL_MULTIPLE * TOL * concentration1),
        inv_softplus(concentration0 - START_TOL_MULTIPLE * TOL * concentration0),
    ]

    concentration1_hat, concentration0_hat = maximum_likelihood_estimate_sgd(
        BetaOutput(),
        samples,
        init_biases=init_biases,
        learning_rate=0.05,
        num_epochs=10,
    )

    print("concentration1:", concentration1_hat, "concentration0:", concentration0_hat)
    assert (
        np.abs(concentration1_hat - concentration1) < TOL * concentration1
    ), f"concentration1 did not match: concentration1 = {concentration1}, concentration1_hat = {concentration1_hat}"
    assert (
        np.abs(concentration0_hat - concentration0) < TOL * concentration0
    ), f"concentration0 did not match: concentration0 = {concentration0}, concentration0_hat = {concentration0_hat}"


@pytest.mark.parametrize("mu_alpha", [(2.5, 0.7)])
def test_neg_binomial(mu_alpha: Tuple[float, float]) -> None:
    """
    Test to check that maximizing the likelihood recovers the parameters
    """
    # test instance
    mu, alpha = mu_alpha

    # generate samples
    mus = torch.zeros((NUM_SAMPLES,)) + mu
    alphas = torch.zeros((NUM_SAMPLES,)) + alpha

    neg_bin_distr = NegativeBinomial(
        total_count=1.0 / alphas, probs=mus * alphas / (1.0 + mus * alphas)
    )
    samples = neg_bin_distr.sample()

    init_biases = [
        inv_softplus(mu - START_TOL_MULTIPLE * TOL * mu),
        inv_softplus(alpha + START_TOL_MULTIPLE * TOL * alpha),
    ]

    mu_hat, alpha_hat = maximum_likelihood_estimate_sgd(
        NegativeBinomialOutput(), samples, init_biases=init_biases, num_epochs=15,
    )

    assert (
        np.abs(mu_hat - mu) < TOL * mu
    ), f"mu did not match: mu = {mu}, mu_hat = {mu_hat}"
    assert (
        np.abs(alpha_hat - alpha) < TOL * alpha
    ), f"alpha did not match: alpha = {alpha}, alpha_hat = {alpha_hat}"


@pytest.mark.parametrize("df, loc, scale,", [(6.0, 2.3, 0.7)])
def test_studentT_likelihood(df: float, loc: float, scale: float):

    dfs = torch.zeros((NUM_SAMPLES,)) + df
    locs = torch.zeros((NUM_SAMPLES,)) + loc
    scales = torch.zeros((NUM_SAMPLES,)) + scale

    distr = StudentT(df=dfs, loc=locs, scale=scales)
    samples = distr.sample()

    init_bias = [
        inv_softplus(df - 2),
        loc - START_TOL_MULTIPLE * TOL * loc,
        inv_softplus(scale - START_TOL_MULTIPLE * TOL * scale),
    ]

    df_hat, loc_hat, scale_hat = maximum_likelihood_estimate_sgd(
        StudentTOutput(),
        samples,
        init_biases=init_bias,
        num_epochs=15,
        learning_rate=1e-3,
    )

    assert (
        np.abs(df_hat - df) < TOL * df
    ), f"df did not match: df = {df}, df_hat = {df_hat}"
    assert (
        np.abs(loc_hat - loc) < TOL * loc
    ), f"loc did not match: loc = {loc}, loc_hat = {loc_hat}"
    assert (
        np.abs(scale_hat - scale) < TOL * scale
    ), f"scale did not match: scale = {scale}, scale_hat = {scale_hat}"


def test_lowrank_multivariate_normal() -> None:
    num_samples = 2000
    dim = 4
    rank = 3

    loc = np.arange(0, dim) / float(dim)
    cov_diag = np.eye(dim) * (np.arange(dim) / dim + 0.5)
    cov_factor = np.sqrt(np.ones((dim, rank)) * 0.2)
    Sigma = cov_factor @ cov_factor.T + cov_diag

    distr = LowRankMultivariateNormal(
        loc=torch.Tensor([loc]),
        cov_diag=torch.Tensor([np.diag(cov_diag)]),
        cov_factor=torch.Tensor([cov_factor]),
    )

    assert np.allclose(
        distr.covariance_matrix.numpy(), Sigma, atol=0.1, rtol=0.1
    ), f"did not match: sigma = {Sigma}, sigma_hat = {distr.variance[0]}"

    samples = distr.sample((num_samples,))

    loc_hat, cov_factor_hat, cov_diag_hat = maximum_likelihood_estimate_sgd(
        LowRankMultivariateNormalOutput(
            dim=dim, rank=rank, sigma_init=0.2, sigma_minimum=0.0
        ),
        samples,
        learning_rate=0.01,
        num_epochs=25,
    )

    distr = LowRankMultivariateNormal(
        loc=torch.Tensor(loc_hat),
        cov_diag=torch.Tensor(cov_diag_hat),
        cov_factor=torch.Tensor(cov_factor_hat),
    )

    Sigma_hat = distr.covariance_matrix.numpy()

    assert np.allclose(
        loc_hat, loc, atol=0.2, rtol=0.1
    ), f"mu did not match: loc = {loc}, loc_hat = {loc_hat}"

    assert np.allclose(
        Sigma_hat, Sigma, atol=0.1, rtol=0.1
    ), f"sigma did not match: sigma = {Sigma}, sigma_hat = {Sigma_hat}"
