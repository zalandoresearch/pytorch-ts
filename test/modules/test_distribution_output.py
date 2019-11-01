import pytest
from typing import Iterable, List, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
from torch.distributions import StudentT

from pts.modules import DistributionOutput, StudentTOutput

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
    learning_rate: float = 1e-2
):
    distr_output.in_features = 1
    arg_proj = distr_output.get_args_proj()
    
    if init_biases is not None:
        for param, bias in zip(arg_proj.proj, init_biases):
            nn.init.constant_(param.bias, bias)
    
    dummy_data = torch.ones((len(samples),1))
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
        return [
            param.detach().numpy() for param in arg_proj(torch.ones((1,1)))
        ]

    return [
        param[0].detach().numpy() for param in arg_proj(torch.ones((1,1)))
    ]



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
        num_epochs=10,
        learning_rate=1e-2
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
