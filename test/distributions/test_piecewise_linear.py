from typing import Tuple, List
import pytest

import torch
import numpy as np

from pts.distributions import PiecewiseLinear

def empirical_cdf(
    samples: np.ndarray, num_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the empirical cdf from the given samples.
    Parameters
    ----------
    samples
        Tensor of samples of shape (num_samples, batch_shape)
    Returns
    -------
    Tensor
        Empirically calculated cdf values. shape (num_bins, batch_shape)
    Tensor
        Bin edges corresponding to the cdf values. shape (num_bins + 1, batch_shape)
    """

    # calculate histogram separately for each dimension in the batch size
    cdfs = []
    edges = []

    batch_shape = samples.shape[1:]
    agg_batch_dim = np.prod(batch_shape, dtype=np.int)

    samples = samples.reshape((samples.shape[0], -1))

    for i in range(agg_batch_dim):
        s = samples[:, i]
        bins = np.linspace(s.min(), s.max(), num_bins + 1)
        hist, edge = np.histogram(s, bins=bins)
        cdfs.append(np.cumsum(hist / len(s)))
        edges.append(edge)

    empirical_cdf = np.stack(cdfs, axis=-1).reshape(num_bins, *batch_shape)
    edges = np.stack(edges, axis=-1).reshape(num_bins + 1, *batch_shape)
    return empirical_cdf, edges

@pytest.mark.parametrize(
    "distr, target, expected_target_cdf, expected_target_crps",
    [
        (
            PiecewiseLinear(
                gamma=torch.ones(size=(1,)),
                slopes=torch.Tensor([2, 3, 1]).reshape(shape=(1, 3)),
                knot_spacings=torch.Tensor([0.3, 0.4, 0.3]).reshape(
                    shape=(1, 3)
                ),
            ),
            [2.2],
            [0.5],
            [0.223000],
        ),
        (
            PiecewiseLinear(
                gamma=torch.ones(size=(2,)),
                slopes=torch.Tensor([[1, 1], [1, 2]]).reshape(shape=(2, 2)),
                knot_spacings=torch.Tensor([[0.4, 0.6], [0.4, 0.6]]).reshape(
                    shape=(2, 2)
                ),
            ),
            [1.5, 1.6],
            [0.5, 0.5],
            [0.083333, 0.145333],
        ),
    ],
)
def test_values(
    distr: PiecewiseLinear,
    target: List[float],
    expected_target_cdf: List[float],
    expected_target_crps: List[float],
):
    target = torch.Tensor(target).reshape(shape=(len(target),))
    expected_target_cdf = np.array(expected_target_cdf).reshape(
        (len(expected_target_cdf),)
    )
    expected_target_crps = np.array(expected_target_crps).reshape(
        (len(expected_target_crps),)
    )

    assert all(np.isclose(distr.cdf(target).numpy(), expected_target_cdf))
    assert all(np.isclose(distr.crps(target).numpy(), expected_target_crps))

    # compare with empirical cdf from samples
    num_samples = 100_000
    samples = distr.sample((num_samples,)).numpy()
    assert np.isfinite(samples).all()

    emp_cdf, edges = empirical_cdf(samples)
    calc_cdf = distr.cdf(torch.Tensor(edges)).numpy()
    assert np.allclose(calc_cdf[1:, :], emp_cdf, atol=1e-2)
