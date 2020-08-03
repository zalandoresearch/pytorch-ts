from typing import Optional, List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        dropout_rate=None,
        bias: bool = True,
        activation=nn.ReLU,
        output_activation: Optional[bool] = None,
    ):
        super().__init__()
        _use_dropout = dropout_rate not in (None, 0)

        num_layers = len(output_sizes)
        _layers = []
        for index in range(num_layers):
            if index == 0:
                in_size = input_size
                out_size = output_sizes[index]
            else:
                in_size = output_sizes[index - 1]
                out_size = output_sizes[index]
            _layers.append(nn.Linear(in_size, out_size, bias=bias))
            if index < (num_layers - 1) or output_activation:
                if _use_dropout:
                    _layers.append(nn.Dropout(dropout_rate))
                _layers.append(activation(inplace=True))

        self._model = nn.Sequential(*_layers)

    def forward(self, inputs):
        return self._model(inputs)
