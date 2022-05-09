from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gluonts.torch.modules.scaler import MeanScaler, NOPScaler
from gluonts.torch.modules.feature import FeatureEmbedder

from pts.feature.lag import get_fourier_lags_for_frequency
from pts.modules import MLP


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


class DNRI_Encoder(nn.Module):
    def __init__(
        self,
        target_dim,
        input_size,
        mlp_hidden_size,
        rnn_hidden_size=None,
        num_edge_types=2,
        save_eval_memory=False,
        cell_type="GRU",
        encoder_mlp_hidden=[],
        prior_mlp_hidden=[],
    ):
        super().__init__()
        self.target_dim = target_dim
        self.num_edge_types = num_edge_types
        self.cell_type = cell_type

        # fully connected graph O(N^2) space
        edges = np.ones(target_dim) - np.eye(target_dim)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.register_buffer(
            "edge2node_mat", torch.Tensor(encode_onehot(self.recv_edges).transpose())
        )

        self.save_eval_memory = save_eval_memory

        self.mlp_1 = MLP(input_size, [mlp_hidden_size, mlp_hidden_size])
        self.mlp_2 = MLP(mlp_hidden_size * 2, [mlp_hidden_size, mlp_hidden_size])
        self.mlp_3 = MLP(mlp_hidden_size, [mlp_hidden_size, mlp_hidden_size])
        self.mlp_4 = MLP(mlp_hidden_size * 3, [mlp_hidden_size, mlp_hidden_size])

        if rnn_hidden_size is None:
            rnn_hidden_size = mlp_hidden_size
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.forward_rnn = rnn(
            input_size=mlp_hidden_size, hidden_size=rnn_hidden_size, batch_first=True
        )
        self.reverse_rnn = rnn(
            input_size=mlp_hidden_size, hidden_size=rnn_hidden_size, batch_first=True
        )

        self.encoder_fc_out = MLP(
            2 * rnn_hidden_size, encoder_mlp_hidden + [num_edge_types]
        )
        self.prior_fc_out = MLP(rnn_hidden_size, prior_mlp_hidden + [num_edge_types])

    def node2edge(self, node_embeddings):
        # Input size: [B, target_dim, T, embed_size]
        send_embed = node_embeddings[:, self.send_edges, ...]
        recv_embed = node_embeddings[:, self.recv_edges, ...]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings):
        old_shape = edge_embeddings.shape
        tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
        incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(
            old_shape[0], -1, old_shape[2], old_shape[3]
        )
        return incoming / (self.target_dim - 1)  # average

    def forward(self, inputs, prior_state=None):
        #  input: [B, T, target_dim, features_per_variate]

        x = inputs.transpose(2, 1).contiguous()  # [B, T, D, F] -> [B, D, T, F]
        x = self.mlp_1(x)  # [B, D, T, F]
        x = self.node2edge(x)  # [B, D^2, T, F*2]
        x = self.mlp_2(x)  # [B, D^2, T, F]
        x_skip = x

        x = self.edge2node(x)  # [B, D^2, T, F] -> [B, D, T, F]
        x = self.mlp_3(x)  # [B, D, T, F]
        x = self.node2edge(x)  # [B, D^2, T, F*2]
        x = torch.cat((x, x_skip), dim=-1)  # [B, D^2, T, F*3]
        x = self.mlp_4(x)  # [B, D^2, T, F]

        old_shape = x.shape
        timesteps = old_shape[2]
        x = x.contiguous().view(-1, timesteps, old_shape[3])  # [B * D^2, T, F]
        forward_x, prior_state = self.forward_rnn(
            x, prior_state
        )  # [B*D^2, T, H], [B*D^2, H]

        prior_logits = (
            self.prior_fc_out(forward_x)
            .view(old_shape[0], old_shape[1], timesteps, self.num_edge_types)
            .transpose(1, 2)
            .contiguous()
        )  # [B, T, D^2 num_edges]

        reverse_x, _ = self.reverse_rnn(x.flip(1))
        combined_x = torch.cat([forward_x, reverse_x.flip(1)], dim=-1)

        posterior_logits = (
            self.encoder_fc_out(combined_x)
            .view(old_shape[0], old_shape[1], timesteps, self.num_edge_types)
            .transpose(1, 2)
            .contiguous()
        )  # [B, T, D^2, num_edges]

        return prior_logits, posterior_logits, prior_state


class DNRI_Decoder(nn.Module):
    def __init__(
        self,
        target_dim,
        input_size,
        decoder_hidden,
        skip_first_edge_type,
        dropout_rate,
        distr_output,
        gumbel_temp,
        num_edge_types=2,
    ):
        super().__init__()

        self.target_dim = target_dim
        self.num_edge_types = num_edge_types
        self.msg_out_shape = decoder_hidden
        self.skip_first_edge_type = skip_first_edge_type
        self.dropout_rate = dropout_rate
        self.gumbel_temp = gumbel_temp

        self.msg_fc1 = nn.ModuleList(
            [
                nn.Linear(2 * decoder_hidden, decoder_hidden)
                for _ in range(num_edge_types)
            ]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(decoder_hidden, decoder_hidden) for _ in range(num_edge_types)]
        )

        self.hidden_r = nn.Linear(decoder_hidden, decoder_hidden, bias=False)
        self.hidden_i = nn.Linear(decoder_hidden, decoder_hidden, bias=False)
        self.hidden_h = nn.Linear(decoder_hidden, decoder_hidden, bias=False)

        self.input_r = nn.Linear(input_size, decoder_hidden, bias=True)
        self.input_i = nn.Linear(input_size, decoder_hidden, bias=True)
        self.input_n = nn.Linear(input_size, decoder_hidden, bias=True)

        self.out_mlp = MLP(decoder_hidden, [decoder_hidden, decoder_hidden])
        self.proj_dist_args = distr_output.get_args_proj(
            decoder_hidden * self.target_dim
        )

        edges = np.ones(target_dim) - np.eye(target_dim)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.register_buffer(
            "edge2node_mat", torch.Tensor(encode_onehot(self.recv_edges))
        )

    def get_initial_hidden(self, inputs_size, device):
        return torch.zeros(
            inputs_size[0], inputs_size[2], self.msg_out_shape, device=device
        )

    def forward(self, inputs, hidden, edge_logits, hard_sample: bool = True):
        old_shape = edge_logits.shape
        edges = F.gumbel_softmax(
            edge_logits.reshape(-1, self.num_edge_types),
            tau=self.gumbel_temp,
            hard=hard_sample,
        ).view(old_shape)

        # node2edge
        receivers = hidden[:, self.recv_edges, ...]
        senders = hidden[:, self.send_edges, ...]

        # pre_msg: [batch, num_edges, 2*msg_out]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = torch.zeros(
            pre_msg.size(0), pre_msg.size(1), self.msg_out_shape, device=inputs.device
        )

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: to exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            if self.training:
                msg = F.dropout(msg, p=self.dropout_rate)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i : i + 1]
            all_msgs += msg / norm

        # This step sums all of the messages per node
        agg_msgs = (
            all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        )
        agg_msgs = agg_msgs.contiguous() / (self.target_dim - 1)  # Average

        # GRU-style gated aggregation
        inp_r = self.input_r(inputs).view(inputs.size(0), self.target_dim, -1)
        inp_i = self.input_i(inputs).view(inputs.size(0), self.target_dim, -1)
        inp_n = self.input_n(inputs).view(inputs.size(0), self.target_dim, -1)
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        pred = self.out_mlp(hidden)
        distr_args = self.proj_dist_args(pred.flatten(1))

        return distr_args, hidden, edges


class DNRIModel(nn.Module):
    def __init__(
        self,
        freq,
        target_dim,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        mlp_hidden_size,
        decoder_hidden,
        skip_first_edge_type,
        dropout_rate,
        distr_output,
        gumbel_temp,
        rnn_hidden_size,
        embedding_dimension: Optional[List[int]] = None,
        lags_seq: Optional[List[int]] = None,
        cell_type="GRU",
        num_edge_types=2,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ):
        super().__init__()

        self.target_dim = target_dim
        self.distr_output = distr_output
        self.target_shape = distr_output.event_shape
        self.num_edge_types = num_edge_types

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_fourier_lags_for_frequency(freq_str=freq)
        self.num_parallel_samples = num_parallel_samples
        self.context_length = context_length
        self.prediction_length = prediction_length

        input_size = self._number_of_features + len(self.lags_seq)

        # input to the encoder has to be [B, T, D, F]
        # but gluonts transformations return [B, T, D*F] -> so need to reshape to [B, T, D, F]
        self.encoder = DNRI_Encoder(
            target_dim=target_dim,  # multivariate dim i.e the  number of nodes: D
            input_size=input_size,  # feature size per variate or feature size per node: F
            mlp_hidden_size=mlp_hidden_size,
            rnn_hidden_size=rnn_hidden_size,
            cell_type=cell_type,
        )

        self.decoder = DNRI_Decoder(
            target_dim=target_dim,
            input_size=input_size,
            decoder_hidden=decoder_hidden,
            skip_first_edge_type=skip_first_edge_type,
            dropout_rate=dropout_rate,
            distr_output=distr_output,
            num_edge_types=num_edge_types,
            gumbel_temp=gumbel_temp,
        )

        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )

        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_static_real
            + self.target_dim  # the log(scale)
            + self.num_feat_dynamic_real
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def unroll_encoder(
        self,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target: Optional[torch.Tensor],
        feat_static_cat: torch.Tensor,
    ):
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, scale = self.scaler(context, observed_context)

        prior_input = past_target[:, : -self.context_length] / scale
        input = (
            torch.cat((context, future_target[:, :-1]), dim=1) / scale
            if future_target is not None
            else context / scale
        )

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, scale.log().squeeze()),
            dim=1,
        )
        expanded_static_feat = (
            static_feat.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, input.shape[1], self.target_dim, -1)
        )

        time_feat = (
            torch.cat(
                (
                    past_time_feat[:, -self.context_length + 1 :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_time_feat is not None
            else past_time_feat[:, -self.context_length + 1 :, ...]
        )
        expanded_time_feat = time_feat.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        features = torch.cat((expanded_static_feat, expanded_time_feat), dim=-1)

        prior_logits, posterior_logits, prior_state, encoder_input = self.unroll(
            prior_input, input, features
        )

        return (
            prior_logits,
            posterior_logits,
            prior_state,
            scale,
            static_feat,
            encoder_input,
        )

        # past_observed_values = torch.min(
        #     past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        # )

        # if future_time_feat is None or future_target is None:
        #     time_feat = past_time_feat[:, -self.context_length :, ...]
        #     sequence = past_target
        #     sequence_length = self.history_length
        #     subsequences_length = self.context_length
        # else:
        #     time_feat = torch.cat(
        #         (
        #             past_time_feat[:, -self.context_length :, ...],
        #             future_time_feat,
        #         ),
        #         dim=1,
        #     )
        #     sequence = torch.cat((past_target, future_target), dim=1)
        #     sequence_length = self.history_length + self.prediction_length
        #     subsequences_length = self.context_length + self.prediction_length

        # # (batch_size, sub_seq_len, target_dim, num_lags)
        # lags = self.get_lagged_subsequences(
        #     sequence=sequence,
        #     sequence_length=sequence_length,
        #     indices=self.lags_seq,
        #     subsequences_length=subsequences_length,
        # )

        # # scale is computed on the context length last units of the past target
        # # scale shape is (batch_size, 1, target_dim)
        # _, scale = self.scaler(
        #     past_target[:, -self.context_length :, ...],
        #     past_observed_values[:, -self.context_length :, ...],
        # )

        # prior_logits, posterior_logits, prior_state, inputs = self.unroll(
        #     lags=lags,
        #     scale=scale,
        #     time_feat=time_feat,
        #     feat_static_cat=feat_static_cat,
        #     feat_static_real=feat_static_real,
        #     unroll_length=subsequences_length,
        # )

        # return prior_logits, posterior_logits, prior_state, scale, inputs

    def unroll(
        self,
        prior_input: torch.Tensor,
        input: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):

        sequence = torch.cat((prior_input, input), dim=1)
        lagged_sequence = self.get_lagged_subsequences(
            sequence=sequence,
            subsequences_length=input.shape[1],
        )

        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(
            lags_shape[0], lags_shape[1], self.target_dim, -1
        )

        if features is None:
            encoder_input = reshaped_lagged_sequence
        else:
            encoder_input = torch.cat((reshaped_lagged_sequence, features), dim=-1)

        prior_logits, posterior_logits, prior_state = self.encoder(
            encoder_input, prior_state=state
        )

        return prior_logits, posterior_logits, prior_state, encoder_input

    # def unroll(
    #     self,
    #     lags: torch.Tensor,
    #     scale: torch.Tensor,
    #     time_feat: torch.Tensor,
    #     feat_static_cat: torch.Tensor,
    #     feat_static_real: torch.Tensor,
    #     unroll_length: int,
    #     prior_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    # ) -> Tuple[
    #     torch.Tensor,
    #     Union[List[torch.Tensor], torch.Tensor],
    #     torch.Tensor,
    #     torch.Tensor,
    # ]:

    #     # (batch_size, sub_seq_len, target_dim, num_lags)
    #     lags_scaled = lags / scale.unsqueeze(-1)

    #     # assert_shape(
    #     #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
    #     # )

    #     # input_lags = lags_scaled.reshape(
    #     #     (-1, unroll_length, len(self.lags_seq) * self.target_dim)
    #     # )

    #     # (batch_size, target_dim, embed_dim)
    #     embedded_cat = self.embedder(feat_static_cat)

    #     # assert_shape(index_embeddings, (-1, self.target_dim, self.embed_dim))

    #     # (batch_size, target_dim, feat_dim)
    #     static_feat = torch.cat(
    #         (
    #             embedded_cat,
    #             # feat_static_real,
    #             scale.log()
    #             if len(self.target_shape) == 0
    #             else scale.squeeze(1).log().unsqueeze(-1),
    #         ),
    #         dim=-1,
    #     )

    #     # (batch_size, seq_len, target_dim, feat_dim)
    #     repeated_static_feat = static_feat.unsqueeze(1).expand(
    #         -1, unroll_length, -1, -1
    #     )

    #     # (batch_size, seq_len, target_dim, time_feat_dim)
    #     repeated_time_feat = time_feat.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

    #     # (batch_size, sub_seq_len, target_dim, input_dim)
    #     inputs = torch.cat(
    #         (lags_scaled, repeated_static_feat, repeated_time_feat), dim=-1
    #     )

    #     prior_logits, posterior_logits, prior_state = self.encoder(
    #         inputs, prior_state=prior_state
    #     )
    #     # unroll encoder
    #     # outputs, state = self.rnn(inputs, begin_state)

    #     # assert_shape(outputs, (-1, unroll_length, self.num_cells))
    #     # for s in state:
    #     #     assert_shape(s, (-1, self.num_cells))

    #     # assert_shape(
    #     #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
    #     # )

    #     return prior_logits, posterior_logits, prior_state, inputs

    def get_lagged_subsequences(
        self,
        sequence: torch.Tensor,
        subsequences_length: int,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        subsequences_length : int
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        sequence_length = sequence.shape[1]
        indices = self.lags_seq

        assert max(indices) + subsequences_length <= sequence_length, (
            "lags cannot go further than history length, found lag"
            f" {max(indices)} while history length is only {sequence_length}"
        )

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    @torch.jit.ignore
    def output_distribution(
        self, params, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, scale=scale)

    # class DNRI_TrainingNetwork(DNRI):
    #     def forward(
    #         self,
    #         target_dimension_indicator: torch.Tensor,
    #         feat_static_cat: torch.Tensor,
    #         feat_static_real: torch.Tensor,
    #         past_time_feat: torch.Tensor,
    #         past_target: torch.Tensor,
    #         past_observed_values: torch.Tensor,
    #         past_is_pad: torch.Tensor,
    #         future_time_feat: torch.Tensor,
    #         future_target: torch.Tensor,
    #         future_observed_values: torch.Tensor,
    #     ):

    #         #  encoder
    #         prior_logits, posterior_logits, _, scale, inputs = self.unroll_encoder(
    #             feat_static_real=feat_static_real,
    #             past_time_feat=past_time_feat,
    #             past_target=past_target,
    #             past_observed_values=past_observed_values,
    #             past_is_pad=past_is_pad,
    #             future_time_feat=future_time_feat,
    #             future_target=future_target,
    #             feat_static_cat=target_dimension_indicator,
    #         )

    #         # decoder
    #         all_distr_args = []
    #         num_time_steps = inputs.size(1)
    #         decoder_hidden = self.decoder.get_initial_hidden(inputs)
    #         for step in range(num_time_steps):
    #             current_inputs = inputs[:, step]
    #             current_p_logits = posterior_logits[:, step]

    #             distr_args, decoder_hidden, _ = self.decoder(
    #                 inputs=current_inputs,
    #                 hidden=decoder_hidden,
    #                 edge_logits=current_p_logits,
    #             )
    #             all_distr_args.append(distr_args)

    #         # loss
    #         # (batch_size, seq_len, target_dim)
    #         target = torch.cat(
    #             (past_target[:, -self.context_length :, ...], future_target),
    #             dim=1,
    #         )

    #         map_stack = partial(torch.stack, dim=1)
    #         all_distr_args = tuple(map(map_stack, zip(*all_distr_args)))

    #         # TODO fix
    #         # distr = self.distr_output.distribution(all_distr_args, scale=scale)
    #         # loss_nll = -distr.log_prob(target).unsqueeze(-1)

    #         # prob = F.softmax(posterior_logits, dim=-1)
    #         # loss_kl = self.kl_categorical_learned(prob, prior_logits)
    #         # loss = loss_nll + self.kl_coef * loss_kl
    #         # return loss.mean()

    # class DNRI_PredictionNetwork(DNRI):
    #     def __init__(self, num_parallel_samples: int, **kwargs) -> None:
    #         super().__init__(**kwargs)
    #         self.num_parallel_samples = num_parallel_samples

    #         # for decoding the lags are shifted by one,
    #         # at the first time-step of the decoder a lag of one corresponds to
    #         # the last target value
    #         self.shifted_lags = [l - 1 for l in self.lags_seq]

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        #  encoder
        prior_logits, _, prior_state, scale, static_feat, inputs = self.unroll_encoder(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat[:, :1],
            future_target=None,
        )

        # decoder_hidden via prior_logits
        decoder_hidden = self.decoder.get_initial_hidden(
            inputs.size(), device=inputs.device
        )
        for k in range(self.context_length):
            params, decoder_hidden, _ = self.decoder(
                inputs[:, k],
                hidden=decoder_hidden,
                edge_logits=prior_logits[:, k],
            )

        # sampling decoder
        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=num_parallel_samples, dim=dim)

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target = repeat(past_target)
        repeated_time_feat = repeat(future_time_feat)
        repeated_time_feat = repeated_time_feat.unsqueeze(2).expand(
            -1, -1, self.target_dim, -1
        )

        repeated_scale = repeat(scale)
        repeated_static_feat = repeat(static_feat).unsqueeze(dim=1).unsqueeze(1)
        repeated_static_feat = repeated_static_feat.expand(-1, 1, self.target_dim, -1)

        repeated_decoder_hidden = repeat(decoder_hidden)
        if self.encoder.cell_type == "LSTM":
            repeated_prior_states = [repeat(s, dim=1) for s in prior_state]
        else:
            repeated_prior_states = repeat(prior_state, dim=1)

        repeated_params = [
            s.repeat_interleave(repeats=self.num_parallel_samples, dim=0).unsqueeze(1)
            for s in params
        ]
        distr = self.output_distribution(repeated_params, scale=repeated_scale)
        next_sample = distr.sample()
        future_samples = [next_sample]

        for k in range(1, self.prediction_length):
            scaled_next_sample = next_sample / repeated_scale
            next_features = torch.cat(
                (repeated_static_feat, repeated_time_feat[:, k : k + 1, ...]),
                dim=-1,
            )
            (prior_logits, _, repeated_prior_states, encoder_input,) = self.unroll(
                repeated_past_target,
                scaled_next_sample,
                next_features,
                state=repeated_prior_states,
            )
            repeated_past_target = torch.cat(
                (repeated_past_target, scaled_next_sample), dim=1
            )

            params, repeated_decoder_hidden, _ = self.decoder(
                # TODO check and fix...
                encoder_input.squeeze(1),
                hidden=repeated_decoder_hidden,
                edge_logits=prior_logits.squeeze(1),
            )
            params = tuple([param.unsqueeze(1) for param in params])
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()
            future_samples.append(next_sample)

            # # older
            # lags = self.get_lagged_subsequences(
            #     sequence=repeated_past_target,
            #     sequence_length=self.history_length + k,
            #     indices=self.shifted_lags,
            #     subsequences_length=1,
            # )

            # current_edge_logits, _, repeated_prior_states, current_inputs = self.unroll(
            #     prior_state=repeated_prior_states,
            #     lags=lags,
            #     scale=repeated_scale,
            #     time_feat=repeated_time_feat[:, k : k + 1, ...],
            #     feat_static_cat=repeated_feat_static_cat,
            #     feat_static_real=repeated_feat_static_real,
            #     unroll_length=1,
            # )

            # distr_args, repeated_decoder_hidden, _ = self.decoder(
            #     current_inputs.squeeze(1),
            #     hidden=repeated_decoder_hidden,
            #     edge_logits=current_edge_logits.squeeze(1),
            # )

            # distr_args = tuple([distr_arg.unsqueeze(1) for distr_arg in distr_args])
            # distr = self.distr_output.distribution(distr_args, scale=repeated_scale)
            # new_samples = distr.sample()

            # future_samples.append(new_samples)
            # repeated_past_target = torch.cat((repeated_past_target, new_samples), dim=1)

        # (batch_size * num_samples, prediction_length, target_dim)
        future_samples_concat = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, target_dim)
        return future_samples_concat.reshape(
            (-1, self.num_parallel_samples, self.prediction_length) + self.target_shape,
        )
