from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gluonts.time_feature import get_seasonality

from pts.model.n_beats.n_beats_network import NBEATSBlock

VALID_N_BEATS_STACK_TYPES = "G", "S", "T"
VALID_LOSS_FUNCTIONS = "sMAPE", "MASE", "MAPE"


class NbeatsXGenericBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
    ):
        super(NbeatsXGenericBlock, self).__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            num_exogenous_time_features=1,  # TODO: clean this hack
        )

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = super().forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)

class NbeatsXNetwork(nn.Module):
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
        super(NbeatsXNetwork, self).__init__()

        self.num_stacks = num_stacks
        self.widths = widths
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.sharing = sharing
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.stack_types = stack_types
        self.prediction_length = prediction_length
        self.context_length = context_length

        self.net_blocks = nn.ModuleList()
        for stack_id in range(num_stacks):
            for block_id in range(num_blocks[stack_id]):
                if self.stack_types[stack_id] == "G":
                    net_block = NbeatsXGenericBlock(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                    )
                else:
                    raise NotImplementedError
                self.net_blocks.append(net_block)

    @staticmethod
    def add_exogenous_variables(past_target, past_time_feat, future_time_feat):
        return torch.cat([past_target, past_time_feat.flatten(start_dim=1), future_time_feat.flatten(start_dim=1)], dim=1)

    def forward(
            self,
            past_time_feat: torch.Tensor,
            past_target: torch.Tensor,
            future_time_feat: torch.Tensor,
    ):
        input_data = self.add_exogenous_variables(past_target, past_time_feat, future_time_feat)
        if len(self.net_blocks) == 1:
            _, forecast = self.net_blocks[0](input_data)
            return forecast
        else:
            backcast, forecast = self.net_blocks[0](input_data)
            backcast = past_target - backcast
            for i in range(1, len(self.net_blocks) - 1):
                b, f = self.net_blocks[i](self.add_exogenous_variables(backcast, past_time_feat, future_time_feat))
                backcast = backcast - b
                forecast = forecast + f
            _, last_forecast = self.net_blocks[-1](self.add_exogenous_variables(backcast, past_time_feat, future_time_feat))
            return forecast + last_forecast

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


class NbeatsXTrainingNetwork(NbeatsXNetwork):
    def __init__(self, loss_function: str, freq: str, *args, **kwargs) -> None:
        super(NbeatsXTrainingNetwork, self).__init__(*args, **kwargs)
        self.loss_function = loss_function
        self.freq = freq

        self.periodicity = get_seasonality(self.freq)

        if self.loss_function == "MASE":
            assert self.periodicity < self.context_length + self.prediction_length, (
                "If the 'periodicity' of your data is less than 'context_length' + 'prediction_length' "
                "the seasonal_error cannot be calculated and thus 'MASE' cannot be used for optimization."
            )

    def forward(
            self,
            past_time_feat: torch.Tensor,
            past_target: torch.Tensor,
            future_target: torch.Tensor,
            future_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        forecast = super().forward(
            past_time_feat=past_time_feat,
            past_target=past_target,
            future_time_feat=future_time_feat,
        )

        if self.loss_function == "sMAPE":
            loss = self.smape_loss(forecast, future_target)
        elif self.loss_function == "MAPE":
            loss = self.mape_loss(forecast, future_target)
        elif self.loss_function == "MASE":
            loss = self.mase_loss(
                forecast, future_target, past_target, self.periodicity
            )
        else:
            raise ValueError(
                f"Invalid value {self.loss_function} for argument loss_function."
            )

        return loss.mean()


class NbeatsXPredictionNetwork(NbeatsXNetwork):
    def __init__(self, *args, **kwargs) -> None:
        super(NbeatsXPredictionNetwork, self).__init__(*args, **kwargs)

    def forward(
            self,
            past_time_feat: torch.Tensor,
            past_target: torch.Tensor,
            future_time_feat: torch.Tensor,
            future_target: torch.Tensor = None
    ) -> torch.Tensor:
        forecasts = super().forward(
            past_time_feat=past_time_feat,
            past_target=past_target,
            future_time_feat=future_time_feat,
        )

        return forecasts.unsqueeze(1)