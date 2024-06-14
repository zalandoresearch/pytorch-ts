from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gluonts.time_feature import get_seasonality

from pts.model.n_beats.base_network import linspace, NBEATSBlock, BaseNbeatsNetwork

VALID_N_BEATS_STACK_TYPES = "G", "S", "T"
VALID_LOSS_FUNCTIONS = "sMAPE", "MASE", "MAPE"


class NBEATSSeasonalBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim=None,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        nb_harmonics=None,
    ):
        if nb_harmonics:
            thetas_dim = nb_harmonics
        else:
            thetas_dim = forecast_length

        super(NBEATSSeasonalBlock, self).__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            share_thetas=True,
        )

        backcast_linspace, forecast_linspace = linspace(
            backcast_length, forecast_length
        )

        p1, p2 = (
            (thetas_dim // 2, thetas_dim // 2)
            if thetas_dim % 2 == 0
            else (thetas_dim // 2, thetas_dim // 2 + 1)
        )
        s1_b = torch.tensor(
            [np.cos(2 * np.pi * i * backcast_linspace) for i in range(p1)]
        ).float()  # H/2-1
        s2_b = torch.tensor(
            [np.sin(2 * np.pi * i * backcast_linspace) for i in range(p2)]
        ).float()
        self.register_buffer("S_backcast", torch.cat([s1_b, s2_b]))

        s1_f = torch.tensor(
            [np.cos(2 * np.pi * i * forecast_linspace) for i in range(p1)]
        ).float()  # H/2-1
        s2_f = torch.tensor(
            [np.sin(2 * np.pi * i * forecast_linspace) for i in range(p2)]
        ).float()
        self.register_buffer("S_forecast", torch.cat([s1_f, s2_f]))

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        backcast = self.theta_b_fc(x).mm(self.S_backcast)
        forecast = self.theta_f_fc(x).mm(self.S_forecast)

        return backcast, forecast


class NBEATSTrendBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        nb_harmonics=None,
    ):
        super(NBEATSTrendBlock, self).__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            share_thetas=True,
        )

        backcast_linspace, forecast_linspace = linspace(
            backcast_length, forecast_length
        )

        self.register_buffer(
            "T_backcast",
            torch.tensor([backcast_linspace ** i for i in range(thetas_dim)]).float(),
        )
        self.register_buffer(
            "T_forecast",
            torch.tensor([forecast_linspace ** i for i in range(thetas_dim)]).float(),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        backcast = self.theta_b_fc(x).mm(self.T_backcast)
        forecast = self.theta_f_fc(x).mm(self.T_forecast)
        return backcast, forecast


class NBEATSGenericBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
    ):
        super(NBEATSGenericBlock, self).__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
        )

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = super().forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)


class NBEATSNetwork(BaseNbeatsNetwork):
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
        super(NBEATSNetwork, self).__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            num_stacks=num_stacks,
            widths=widths,
            num_blocks=num_blocks,
            num_block_layers=num_block_layers,
            expansion_coefficient_lengths=expansion_coefficient_lengths,
            sharing=sharing,
            stack_types=stack_types,
            **kwargs,
        )

        self.net_blocks = nn.ModuleList()
        for stack_id in range(num_stacks):
            for block_id in range(num_blocks[stack_id]):
                if self.stack_types[stack_id] == "G":
                    net_block = NBEATSGenericBlock(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                    )
                elif self.stack_types[stack_id] == "S":
                    net_block = NBEATSSeasonalBlock(
                        units=self.widths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                    )
                else:
                    net_block = NBEATSTrendBlock(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                    )
                self.net_blocks.append(net_block)

    def forward(self, past_target: torch.Tensor):
        if len(self.net_blocks) == 1:
            _, forecast = self.net_blocks[0](past_target)
            return forecast
        else:
            backcast, forecast = self.net_blocks[0](past_target)
            backcast = past_target - backcast
            for i in range(1, len(self.net_blocks) - 1):
                b, f = self.net_blocks[i](backcast)
                backcast = backcast - b
                forecast = forecast + f
            _, last_forecast = self.net_blocks[-1](backcast)
            return forecast + last_forecast


class NBEATSTrainingNetwork(NBEATSNetwork):
    def __init__(self, loss_function: str, freq: str, *args, **kwargs) -> None:
        super(NBEATSTrainingNetwork, self).__init__(*args, **kwargs)
        self.loss_function = loss_function
        self.freq = freq

        self.periodicity = get_seasonality(self.freq)

        if self.loss_function == "MASE":
            assert self.periodicity < self.context_length + self.prediction_length, (
                "If the 'periodicity' of your data is less than 'context_length' + 'prediction_length' "
                "the seasonal_error cannot be calculated and thus 'MASE' cannot be used for optimization."
            )

    def forward(
        self, past_target: torch.Tensor, future_target: torch.Tensor
    ) -> torch.Tensor:
        forecast = super().forward(past_target=past_target)

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


class NBEATSPredictionNetwork(NBEATSNetwork):
    def __init__(self, *args, **kwargs) -> None:
        super(NBEATSPredictionNetwork, self).__init__(*args, **kwargs)

    def forward(
        self, past_target: torch.Tensor, future_target: torch.Tensor = None
    ) -> torch.Tensor:
        forecasts = super().forward(past_target=past_target)

        return forecasts.unsqueeze(1)
