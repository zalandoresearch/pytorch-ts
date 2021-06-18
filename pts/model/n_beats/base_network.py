from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


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


class BaseNbeatsNetwork(nn.Module):

    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        num_stacks: int,
        widths: List[int],
        num_blocks: List[int],
        num_block_layers: List[int],
        expansion_coefficient_lengths: List[int],
        sharing: List[bool],
        stack_types: List[str],
        **kwargs,
    ) -> None:
        super(BaseNbeatsNetwork, self).__init__()

        self.num_stacks = num_stacks
        self.widths = widths
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.sharing = sharing
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.stack_types = stack_types
        self.prediction_length = prediction_length
        self.context_length = context_length

    def forward(self, **kwargs):
        raise NotImplementedError

    def smape_loss(
        self, forecast: torch.Tensor, future_target: torch.Tensor
    ) -> torch.Tensor:
        denominator = (torch.abs(future_target) + torch.abs(forecast)).detach()
        flag = denominator == 0

        return (200 / self.prediction_length) * torch.mean(
            (torch.abs(future_target - forecast) * torch.logical_not(flag))
            / (denominator + flag),
            dim=1,
        )

    def mape_loss(
        self, forecast: torch.Tensor, future_target: torch.Tensor
    ) -> torch.Tensor:
        denominator = torch.abs(future_target)
        flag = denominator == 0

        return (100 / self.prediction_length) * torch.mean(
            (torch.abs(future_target - forecast) * torch.logical_not(flag))
            / (denominator + flag),
            dim=1,
        )

    def mase_loss(
        self,
        forecast: torch.Tensor,
        future_target: torch.Tensor,
        past_target: torch.Tensor,
        periodicity: int,
    ) -> torch.Tensor:
        factor = 1 / (self.context_length + self.prediction_length - periodicity)

        whole_target = torch.cat((past_target, future_target), dim=1)
        seasonal_error = factor * torch.mean(
            torch.abs(
                whole_target[:, periodicity:, ...]
                - whole_target[:, :-periodicity:, ...]
            ),
            dim=1,
        )
        flag = seasonal_error == 0

        return (
            torch.mean(torch.abs(future_target - forecast), dim=1)
            * torch.logical_not(flag)
        ) / (seasonal_error + flag)
