from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gluonts.time_feature import get_seasonality

from pts.model.n_beats.base_network import NBEATSBlock, BaseNbeatsNetwork

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
        num_feat_dynamic_real=0,
    ):
        super(NbeatsXGenericBlock, self).__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            num_feat_dynamic_real=num_feat_dynamic_real,
        )
        self.num_feat_dynamic_real = num_feat_dynamic_real

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = super().forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)


class NbeatsXNetwork(BaseNbeatsNetwork):
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
        num_feat_dynamic_real: int,
        **kwargs,
    ) -> None:
        super(NbeatsXNetwork, self).__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            num_stacks=num_stacks,
            widths=widths,
            num_blocks=num_blocks,
            num_block_layers=num_block_layers,
            expansion_coefficient_lengths=expansion_coefficient_lengths,
            sharing=sharing,
            stack_types=stack_types,
            **kwargs
        )

        self.num_feat_dynamic_real = num_feat_dynamic_real

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
                        num_feat_dynamic_real=num_feat_dynamic_real,
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