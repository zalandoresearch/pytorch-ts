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
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[self.cell_type]
        self.rnn = rnn(input_size=1,
                       hidden_size=num_cells,
                       num_layers=num_layers,
                       dropout=dropout_rate,
                       batch_first=True)

        # TODO
        # self.target_shape = distr_output.event_shape

        self.proj_distr_args = distr_output.get_args_proj()

        self.embedder = FeatureEmbedder(cardinalities=cardinality,
                                        embedding_dims=embedding_dimension)

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
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length : int
            length of sequence in the T (time) dimension (axis = 1).
        indices : List[int]
            list of lag indices to be used.
        subsequences_length : int
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}")
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    @staticmethod
    def weighted_average(tensor: torch.Tensor,
                         weights: Optional[torch.Tensor] = None,
                         dim=None):
        if weights is not None:
            weighted_tensor = tensor * weights
            sum_weights = torch.max(torch.ones_like(weights.sum(dim=dim)),
                                    weights.sum(dim=dim))
            return weighted_tensor.sum(dim=dim) / sum_weights
        else:
            return tensor.mean(dim=dim)

    def unroll_encoder(
            self,
            feat_static_cat: torch.Tensor,  # (batch_size, num_features)
            feat_static_real: torch.Tensor,  # (batch_size, num_features)
            past_time_feat: torch.Tensor,  # (batch_size, history_length, num_features)
            past_target: torch.Tensor,  # (batch_size, history_length, *target_shape)
            past_observed_values: torch.Tensor,  # (batch_size, history_length, *target_shape)
            future_time_feat: Optional[
                torch.Tensor]=None,  # (batch_size, prediction_length, num_features)
            future_target: Optional[
                torch.Tensor]=None,  # (batch_size, prediction_length, *target_shape)
    ) -> Tuple[torch.Tensor, List, torch.Tensor, torch.Tensor]:
        
        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat[:,self.history_length - self.context_length:,...]
            sequence = past_target
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (
                    past_time_feat[:,self.history_length - self.context_length:,...],
                    future_time_feat,
                ),
                dim=1)
            sequence = torch.cat((past_target, future_target), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length
        
        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length)

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        _, scale = self.scaler(
            past_target[:,self.context_length:,...],
            past_observed_values[:,self.context_length:,...]
        )

        # (batch_size, num_features)
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help
        # prediction too
        # (batch_size, num_features + prod(target_shape))
        static_feat = torch.cat((
            embedded_cat,
            feat_static_real,
            scale.log()
            if len(self.target_shape) == 0
            else scale.squeeze(1).log()
        ), dim=1)

        # (batch_size, subsequences_length, num_features + 1)
        repeated_static_feat = static_feat.expand(-1, subsequences_length, -1)

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = lags_scaled.reshape((-1, subsequences_length, len(self.lags_seq) * prod(self.target_shape)))

        # unroll encoder
        outputs, state = self.rnn(inputs)

        # outputs: (batch_size, seq_len, num_cells)
        # state: list of (batch_size, num_cells) tensors
        # scale: (batch_size, 1, *target_shape)
        # static_feat: (batch_size, num_features + prod(target_shape))
        return outputs, state, scale, static_feat

class DeepARTrainingNetwork(DeepARNetwork):
    pass