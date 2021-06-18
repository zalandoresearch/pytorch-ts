from typing import Tuple

import numpy as np
from torch import nn as nn


def linspace(
    backcast_length: int, forecast_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    lin_space = np.linspace(
        -backcast_length,
        forecast_length,
        backcast_length + forecast_length,
        dtype=np.float32,
    )
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class NBEATSBlock(nn.Module):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        share_thetas=False,
        num_exogenous_time_features=0,
    ):
        super(NBEATSBlock, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.num_exogenous_time_features = num_exogenous_time_features
        self.input_dim = backcast_length +  num_exogenous_time_features * (backcast_length + forecast_length)

        fc_stack = [nn.Linear(self.input_dim, units), nn.ReLU()]
        for _ in range(num_block_layers - 1):
            fc_stack.append(nn.Linear(units, units))
            fc_stack.append(nn.ReLU())
        self.fc = nn.Sequential(*fc_stack)

        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        return self.fc(x)