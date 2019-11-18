from typing import List

import torch
import torch.nn as nn

import numpy as np

from pts.modules import DistributionOutput, MeanScaler, NOPScaler, FeatureEmbedder


class DeepARNetwork(nn.Module):
    def __init__(
            self,
            num_layers: int,
            num_cells: int,
            cell_type: str,
            history_length: int,
            context_length: int,
            prediction_length: int,
            distr_output: DistributionOutput,
            dropout_rate: float,
            cardinality: List[int],
            embedding_dimension: List[int],
            lags_seq: List[int],
            scaling: bool = True,
            dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)
        self.scaling = scaling
        self.dtype = dtype

        self.lags_seq = lags_seq

        self.distr_output = distr_output
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[
            self.cell_type
        ]
        self.rnn = rnn(
            input_size=1,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True
        )

        # TODO
        # self.target_shape = distr_output.event_shape

        self.proj_distr_args = distr_output.get_args_proj()

        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=embedding_dimension
        )

        if scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:

        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(
                sequence[:,begin_index:end_index,...]
            )
        return torch.stack(lagged_values, dim=-1)


class DeepARTrainingNetwork(DeepARNetwork):
    pass