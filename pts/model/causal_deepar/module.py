from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import (
    lagged_sequence_values,
    repeat_along_dim,
    unsqueeze_expand,
)
from gluonts.itertools import prod


class CausalDeepARModel(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        embedding_dimension: Optional[List[int]] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        input_size: int = 1,
        control_size: int = 1,
        distr_output: DistributionOutput = StudentTOutput(),
        control_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()

        assert distr_output.event_shape == ()

        self.context_length = context_length
        self.prediction_length = prediction_length

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.num_parallel_samples = num_parallel_samples

        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        self.input_size = input_size
        self.control_size = control_size
        if scaling:
            self.scaler = MeanScaler(dim=-1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=-1, keepdim=True)
        self.control_scaler = NOPScaler(dim=-1, keepdim=True)

        rnn_input_size = (
            len(self.lags_seq) * self.input_size
            + len(self.lags_seq) * self.control_size
            + self._number_of_features
        )

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.distr_output = distr_output
        self.control_output = control_output
        self.param_proj = distr_output.get_args_proj(hidden_size + self.control_size)
        self.param_proj_control = control_output.get_args_proj(hidden_size)

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + self.input_size  # the log(input scale)
            + self.control_size  # the log(control scale)
        )

    def input_shapes(self, batch_size=1) -> Dict[str, Tuple[int, ...]]:
        return {
            "feat_static_cat": (batch_size, self.num_feat_static_cat),
            "feat_static_real": (batch_size, self.num_feat_static_real),
            "past_time_feat": (
                batch_size,
                self._past_length,
                self.num_feat_dynamic_real,
            ),
            "past_target": (batch_size, self._past_length),
            "past_control": (batch_size, self._past_length),
            "past_observed_values": (batch_size, self._past_length),
            "future_time_feat": (
                batch_size,
                self.prediction_length,
                self.num_feat_dynamic_real,
            ),
        }

    def input_types(self) -> Dict[str, torch.dtype]:
        return {
            "feat_static_cat": torch.long,
            "feat_static_real": torch.float,
            "past_time_feat": torch.float,
            "past_target": torch.float,
            "past_control": torch.float,
            "past_observed_values": torch.float,
            "future_time_feat": torch.float,
        }

    def unroll_lagged_rnn(
        self,
        feat_static_cat: torch.Tensor,  # (batch_size, num_features)
        feat_static_real: torch.Tensor,  # (batch_size, num_features)
        past_time_feat: torch.Tensor,  # (batch_size, history_length, num_features)
        past_target: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_control: torch.Tensor,  # (batch_size, history_length, *target_shape)
        future_control: Optional[torch.Tensor] = None,
        future_time_feat: Optional[
            torch.Tensor
        ] = None,  # (batch_size, prediction_length, num_features)
        future_target: Optional[
            torch.Tensor
        ] = None,  # (batch_size, prediction_length, *target_shape)
    ) -> Tuple[
        Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:

        context = past_target[..., -self.context_length :]
        observed_context = past_observed_values[..., -self.context_length :]
        _, scale = self.scaler(context, observed_context)

        control = past_control[..., -self.context_length :]
        _, control_scale = self.control_scaler(control, observed_context)

        prior_input = past_target[..., : -self.context_length] / scale
        prior_control = past_control[..., : -self.context_length] / control_scale

        input = (
            torch.cat((context, future_target[..., :-1]), dim=-1) / scale
            if future_target is not None
            else context / scale
        )

        input_control = (
            torch.cat((control, future_control[..., :-1]), dim=-1) / control_scale
            if future_control is not None
            else control / control_scale
        )

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, scale.log()),
            dim=-1,
        )
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=-2, size=input.shape[-1]
        )

        time_feat = (
            torch.cat(
                (
                    past_time_feat[..., -self.context_length + 1 :, :],
                    future_time_feat,
                ),
                dim=-2,
            )
            if future_time_feat is not None
            else past_time_feat[..., -self.context_length + 1 :, :]
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)
        lags = lagged_sequence_values(self.lags_seq, prior_input, input, dim=-1)
        lags_control = lagged_sequence_values(
            self.lags_seq, prior_control, input_control, dim=-1
        )

        rnn_input = torch.cat((lags, lags_control, features), dim=-1)
        output, new_state = self.rnn(rnn_input)

        control_params = self.param_proj_control(output)

        params = self.param_proj(
            torch.cat((output, input_control.unsqueeze(-1)), dim=-1)
        )

        # outputs: (batch_size, seq_len, num_cells)
        # state: list of (num_layers, batch_size, num_cells) tensors
        # scale: (batch_size, 1, *target_shape)
        # static_feat: (batch_size, num_features + prod(target_shape))
        return (
            params,
            control_params,
            scale,
            control_scale,
            output,
            static_feat,
            new_state,
        )

    @torch.jit.ignore
    def output_distribution(
        self, params, control_params, scale=None, control_scale=None, trailing_n=None
    ) -> Tuple[torch.distributions.Distribution, torch.distributions.Distribution]:
        """
        Instantiate the output distribution

        Parameters
        ----------
        params
            Tuple of distribution parameters.
        scale
            (Optional) scale tensor.
        trailing_n
            If set, the output distribution is created only for the last
            ``trailing_n`` time points.

        Returns
        -------
        torch.distributions.Distribution
            Output distribution from the model.
        """
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]

        sliced_control_params = control_params
        if trailing_n is not None:
            sliced_control_params = [p[:, -trailing_n:] for p in control_params]

        return self.distr_output.distribution(
            sliced_params, scale=scale
        ), self.control_output.distribution(sliced_control_params, scale=control_scale)

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_control: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_control: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Invokes the model on input data, and produce outputs future samples.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            (Optional) tensor of dynamic real features in the past,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        num_parallel_samples
            How many future samples to produce.
            By default, self.num_parallel_samples is used.
        """
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        params, scale, _, static_feat, state = self.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )

        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            / repeated_scale
        )
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_state = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=1) for s in state
        ]

        repeated_params = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=0) for s in params
        ]
        distr = self.output_distribution(
            repeated_params, trailing_n=1, scale=repeated_scale
        )
        next_sample = distr.sample()
        future_samples = [next_sample]

        for k in range(1, self.prediction_length):
            scaled_next_sample = next_sample / repeated_scale
            next_features = torch.cat(
                (repeated_static_feat, repeated_time_feat[:, k : k + 1]),
                dim=-1,
            )
            next_lags = lagged_sequence_values(
                self.lags_seq, repeated_past_target, scaled_next_sample, dim=-1
            )
            rnn_input = torch.cat((next_lags, next_features), dim=-1)

            output, repeated_state = self.rnn(rnn_input, repeated_state)

            repeated_past_target = torch.cat(
                (repeated_past_target, scaled_next_sample), dim=1
            )

            params = self.param_proj(output)
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()
            future_samples.append(next_sample)

        future_samples_concat = torch.cat(future_samples, dim=1)

        return future_samples_concat.reshape(
            (-1, num_parallel_samples, self.prediction_length)
        )

    def log_prob(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_control: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_control: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
    ) -> torch.Tensor:
        return -self.loss(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_control=past_control,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_control=future_control,
            future_target=future_target,
            future_observed_values=torch.ones_like(future_target),
            loss=NegativeLogLikelihood(),
            future_only=True,
            aggregate_by=torch.sum,
        )

    def loss(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_control: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_control: torch.Tensor,
        future_observed_values: torch.Tensor,
        loss: DistributionLoss = NegativeLogLikelihood(),
        future_only: bool = False,
        aggregate_by=torch.mean,
    ) -> torch.Tensor:
        extra_dims = len(future_target.shape) - len(past_target.shape)
        extra_shape = future_target.shape[:extra_dims]
        batch_shape = future_target.shape[: extra_dims + 1]

        repeats = prod(extra_shape)
        feat_static_cat = repeat_along_dim(feat_static_cat, 0, repeats)
        feat_static_real = repeat_along_dim(feat_static_real, 0, repeats)
        past_time_feat = repeat_along_dim(past_time_feat, 0, repeats)
        past_target = repeat_along_dim(past_target, 0, repeats)
        past_control = repeat_along_dim(past_control, 0, repeats)
        past_observed_values = repeat_along_dim(past_observed_values, 0, repeats)
        future_time_feat = repeat_along_dim(future_time_feat, 0, repeats)

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        )
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        )
        future_control_reshaped = future_control.reshape(
            -1,
            *future_control.shape[extra_dims + 1 :],
        )

        # params, control_params, scale, control_scale, output, static_feat, new_state
        params, control_params, scale, control_scale, _, _, _ = self.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            past_control,
            future_control_reshaped,
            future_time_feat,
            future_target_reshaped,
        )

        if future_only:
            distr, control_distr = self.output_distribution(
                params, scale, trailing_n=self.prediction_length
            )
            loss_values_target = (
                loss(distr, future_target_reshaped) * future_observed_reshaped
            )
            loss_values_control = (
                loss(control_distr, future_control_reshaped) * future_observed_reshaped
            )
            loss_values = loss_values_target + loss_values_control

        else:
            distr = self.output_distribution(params, scale)
            control_distr = self.control_output_distribution(
                control_params, control_scale
            )

            context_target = past_target[:, -self.context_length + 1 :]
            context_control = past_control[:, -self.context_length + 1 :]
            target = torch.cat(
                (context_target, future_target_reshaped),
                dim=1,
            )
            control = torch.cat(
                (context_control, future_control_reshaped),
                dim=1,
            )
            context_observed = past_observed_values[:, -self.context_length + 1 :]
            observed_values = torch.cat(
                (context_observed, future_observed_reshaped), dim=1
            )
            loss_values_target = loss(distr, target) * observed_values
            loss_values_control = loss(control_distr, control) * observed_values
            loss_values = loss_values_target + loss_values_control

        loss_values = loss_values.reshape(*batch_shape, *loss_values.shape[1:])

        return aggregate_by(
            loss_values,
            dim=tuple(range(extra_dims + 1, len(future_target.shape))),
        )


class CausalDeepARPredictionNetwork(CausalDeepARNetwork):
    def __init__(self, num_parallel_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one, at the first time-step
        # of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        static_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_control: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        control_scale: torch.Tensor,
        future_control: torch.Tensor,
        begin_states: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        static_feat : Tensor
            static features. Shape: (batch_size, num_static_features).
        past_target : Tensor
            target history. Shape: (batch_size, history_length).
        time_feat : Tensor
            time features. Shape: (batch_size, prediction_length, num_time_features).
        scale : Tensor
            tensor containing the scale of each element in the batch. Shape: (batch_size, 1, 1).
        begin_states : List or Tensor
            list of initial states for the LSTM layers or tensor for GRU.
            the shape of each tensor of the list should be (num_layers, batch_size, num_cells)
        Returns
        --------
        Tensor
            A tensor containing sampled paths.
            Shape: (batch_size, num_sample_paths, prediction_length).
        """

        # blows-up the dimension of each tensor to batch_size * self.num_parallel_samples for increasing parallelism
        repeated_past_target = past_target.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_time_feat = time_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_past_control = past_control.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        ).unsqueeze(1)
        repeated_future_control = future_control.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_control_scale = control_scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        if self.cell_type == "LSTM":
            repeated_states = [
                s.repeat_interleave(repeats=self.num_parallel_samples, dim=1)
                for s in begin_states
            ]
        else:
            repeated_states = begin_states.repeat_interleave(
                repeats=self.num_parallel_samples, dim=1
            )

        future_samples = []

        # for each future time-units we draw new samples for this time-unit and update the state
        for k in range(self.prediction_length):
            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )
            control_lags = self.get_lagged_subsequences(
                sequence=repeated_past_control,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags_scaled = lags / repeated_scale.unsqueeze(-1)

            control_lags_scaled = control_lags / repeated_control_scale.unsqueeze(-1)

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = lags_scaled.reshape(
                (-1, 1, prod(self.target_shape) * len(self.lags_seq))
            )

            input_control_lags = control_lags_scaled.reshape(
                (-1, 1, len(self.lags_seq) * prod(self.target_shape))
            )

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags + num_time_features + num_static_features)
            decoder_input = torch.cat(
                (
                    input_lags,
                    input_control_lags,
                    repeated_time_feat[:, k : k + 1, :],
                    repeated_static_feat,
                ),
                dim=-1,
            )

            # output shape: (batch_size * num_samples, 1, num_cells)
            # state shape: (batch_size * num_samples, num_cells)
            rnn_outputs, repeated_states = self.rnn(decoder_input, repeated_states)

            control_dist_args = self.proj_control_args(rnn_outputs)
            control_distr = self.control_output.distribution(
                control_dist_args, repeated_control_scale
            )

            new_control_sample = control_distr.sample()
            control = repeated_future_control[:, k : k + 1]
            control[control != control] = new_control_sample[control != control]
            repeated_past_control = torch.cat((repeated_past_control, control), dim=1)

            distr_args = self.proj_distr_args(
                torch.cat((rnn_outputs, control.unsqueeze(-1)), dim=-1)
            )

            # compute likelihood of target given the predicted parameters
            distr = self.distr_output.distribution(distr_args, scale=repeated_scale)

            # (batch_size * num_samples, 1, *target_shape)
            new_samples = distr.sample()

            # (batch_size * num_samples, seq_len, *target_shape)
            repeated_past_target = torch.cat((repeated_past_target, new_samples), dim=1)
            future_samples.append(new_samples)

        # (batch_size * num_samples, prediction_length, *target_shape)
        samples = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, *target_shape)
        return samples.reshape(
            (
                (-1, self.num_parallel_samples)
                + (self.prediction_length,)
                + self.target_shape
            )
        )

    # noinspection PyMethodOverriding,PyPep8Naming
    def forward(
        self,
        feat_static_cat: torch.Tensor,  # (batch_size, num_features)
        feat_static_real: torch.Tensor,  # (batch_size, num_features)
        past_time_feat: torch.Tensor,  # (batch_size, history_length, num_features)
        past_target: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: torch.Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: torch.Tensor,  # (batch_size, prediction_length, num_features)
        past_control: torch.Tensor,
        future_control: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts samples, all tensors should have NTC layout.
        Parameters
        ----------
        feat_static_cat : (batch_size, num_features)
        feat_static_real : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)

        Returns
        -------
        Tensor
            Predicted samples
        """

        # unroll the decoder in "prediction mode", i.e. with past data only
        _, state, scale, control_scale, static_feat = self.unroll_encoder(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_control=past_control,
            future_time_feat=None,
            future_target=None,
            future_control=None,
        )

        return self.sampling_decoder(
            past_target=past_target,
            past_control=past_control,
            time_feat=future_time_feat,
            static_feat=static_feat,
            future_control=future_control,
            scale=scale,
            control_scale=control_scale,
            begin_states=state,
        )
