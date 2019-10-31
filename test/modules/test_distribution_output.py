import pytest
from typing import Iterable, List, Tuple

import torch

from pts.modules import DistributionOutput, StudentTOutput

NUM_SAMPLES = 2000
BATCH_SIZE = 32
TOL = 0.3
START_TOL_MULTIPLE = 1

def maximum_likelihood_estimate_sgd(
    distr_output: DistributionOutput,
    samples: torch.Tensor,
    init_biases: List[torch.Tensor] = None,
    num_epochs: int = 5,
    learning_rate: float = 1e-2
):
    device = torch.device("cpu")

    arg_proj = distr_output.get_args_proj()
    arg_proj.initialize()


@pytest.mark.parametrize("loc, scale, df", [(2.3, 0.7, 6.0)])
def test_studentT_likelihood():
    pass