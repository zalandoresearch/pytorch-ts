from typing import List, Optional, Callable, Tuple, Union
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pts.modules import MLP
from pts.model import weighted_average
from pts.modules import DistributionOutput, MeanScaler, NOPScaler, FeatureEmbedder


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
        #  [B, T, target_dim, input_size]
        x = inputs.transpose(2, 1).contiguous()
        x = self.mlp_1(x)
        x = self.node2edge(x)
        x = self.mlp_2(x)
        x_skip = x

        x = self.edge2node(x)
        x = self.mlp_3(x)
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=-1)
        x = self.mlp_4(x)

        old_shape = x.shape
        timesteps = old_shape[2]
        x = x.contiguous().view(-1, timesteps, old_shape[3])
        forward_x, prior_state = self.forward_rnn(x, prior_state)
        reverse_x, _ = self.reverse_rnn(x.flip(1))
        combined_x = torch.cat([forward_x, reverse_x.flip(1)], dim=-1)

        posterior_logits = (
            self.encoder_fc_out(combined_x)
            .view(old_shape[0], old_shape[1], timesteps, self.num_edge_types)
            .transpose(1, 2)
            .contiguous()
        )

        prior_logits = (
            self.prior_fc_out(forward_x)
            .view(old_shape[0], old_shape[1], timesteps, self.num_edge_types)
            .transpose(1, 2)
            .contiguous()
        )

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

    def get_initial_hidden(self, inputs):
        return torch.zeros(
            inputs.size(0), inputs.size(2), self.msg_out_shape, device=inputs.device
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


class DNRI(nn.Module):
    def __init__(
        self,
        target_dim,
        input_size,
        embedding_dimension,
        lags_seq,
        mlp_hidden_size,
        decoder_hidden,
        skip_first_edge_type,
        dropout_rate,
        history_length,
        context_length,
        prediction_length,
        distr_output,
        gumbel_temp,
        kl_coef,
        rnn_hidden_size,
        cell_type="GRU",
        num_edge_types=2,
        scaling: bool = True,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.distr_output = distr_output
        self.target_shape = distr_output.event_shape
        self.num_edge_types = num_edge_types
        self.kl_coef = kl_coef
        assert len(set(lags_seq)) == len(lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()
        self.lags_seq = lags_seq

        self.encoder = DNRI_Encoder(
            target_dim=target_dim,
            input_size=input_size,
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

        if scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.embed = nn.Embedding(target_dim, embedding_dimension)

        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length

    def unroll_encoder(
        self,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
        feat_static_cat: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length :, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (past_time_feat[:, -self.context_length :, ...], future_time_feat,),
                dim=1,
            )
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, target_dim)
        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )

        prior_logits, posterior_logits, prior_state, inputs = self.unroll(
            lags=lags,
            scale=scale,
            time_feat=time_feat,
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            unroll_length=subsequences_length,
        )

        return prior_logits, posterior_logits, prior_state, scale, inputs

    def unroll(
        self,
        lags: torch.Tensor,
        scale: torch.Tensor,
        time_feat: torch.Tensor,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        unroll_length: int,
        prior_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:

        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)

        # assert_shape(
        #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
        # )

        # input_lags = lags_scaled.reshape(
        #     (-1, unroll_length, len(self.lags_seq) * self.target_dim)
        # )

        # (batch_size, target_dim, embed_dim)
        embedded_cat = self.embed(feat_static_cat)

        # assert_shape(index_embeddings, (-1, self.target_dim, self.embed_dim))

        # (batch_size, target_dim, feat_dim)
        static_feat = torch.cat(
            (
                embedded_cat,
                # feat_static_real,
                scale.log()
                if len(self.target_shape) == 0
                else scale.squeeze(1).log().unsqueeze(-1),
            ),
            dim=-1,
        )

        # (batch_size, seq_len, target_dim, feat_dim)
        repeated_static_feat = static_feat.unsqueeze(1).expand(
            -1, unroll_length, -1, -1
        )

        # (batch_size, seq_len, target_dim, time_feat_dim)
        repeated_time_feat = time_feat.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        # (batch_size, sub_seq_len, target_dim, input_dim)
        inputs = torch.cat(
            (lags_scaled, repeated_static_feat, repeated_time_feat), dim=-1
        )

        prior_logits, posterior_logits, prior_state = self.encoder(
            inputs, prior_state=prior_state
        )
        # unroll encoder
        # outputs, state = self.rnn(inputs, begin_state)

        # assert_shape(outputs, (-1, unroll_length, self.num_cells))
        # for s in state:
        #     assert_shape(s, (-1, self.num_cells))

        # assert_shape(
        #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
        # )

        return prior_logits, posterior_logits, prior_state, inputs

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
        sequence
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length
            length of sequence in the T (time) dimension (axis = 1).
        indices
            list of lag indices to be used.
        subsequences_length
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences.
            Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: history_length + begin_index >= 0
        # that is: history_length - lag_index - sequence_length >= 0
        # hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)


class DNRI_TrainingNetwork(DNRI):
    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
    ):

        #  encoder
        prior_logits, posterior_logits, _, scale, inputs = self.unroll_encoder(
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            feat_static_cat=target_dimension_indicator,
        )

        # decoder
        all_distr_args = []
        num_time_steps = inputs.size(1)
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        for step in range(num_time_steps):
            current_inputs = inputs[:, step]
            current_p_logits = posterior_logits[:, step]

            distr_args, decoder_hidden, _ = self.decoder(
                inputs=current_inputs,
                hidden=decoder_hidden,
                edge_logits=current_p_logits,
            )
            all_distr_args.append(distr_args)

        # loss
        # (batch_size, seq_len, target_dim)
        target = torch.cat(
            (past_target_cdf[:, -self.context_length :, ...], future_target_cdf), dim=1,
        )

        map_stack = partial(torch.stack, dim=1)
        all_distr_args = tuple(map(map_stack, zip(*all_distr_args)))
        distr = self.distr_output.distribution(all_distr_args, scale=scale)
        loss_nll = -distr.log_prob(target).unsqueeze(-1)

        prob = F.softmax(posterior_logits, dim=-1)
        loss_kl = self.kl_categorical_learned(prob, prior_logits)
        loss = loss_nll + self.kl_coef * loss_kl
        return loss.mean()

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds * (torch.log(preds + 1e-16) - log_prior)
        # if self.normalize_kl:
        #     return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        # elif self.normalize_kl_per_var:
        #     return kl_div.sum() / (self.target_dim * preds.size(0))
        # else:
        return kl_div.view(preds.size(0), preds.size(1), -1).sum(dim=-1, keepdims=True)


class DNRI_PredictionNetwork(DNRI):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to
        # the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:

        #  encoder
        prior_logits, _, prior_state, scale, inputs = self.unroll_encoder(
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
            feat_static_cat=target_dimension_indicator,
        )

        # decoder_hidden via prior_logits
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        for k in range(self.context_length):
            current_inputs = inputs[:, k]
            current_edge_logits = prior_logits[:, k]
            _, decoder_hidden, _ = self.decoder(
                current_inputs, hidden=decoder_hidden, edge_logits=current_edge_logits,
            )

        # sampling decoder
        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(future_time_feat)
        repeated_scale = repeat(scale)
        repeated_feat_static_cat = repeat(target_dimension_indicator)
        repeated_feat_static_real = repeat(feat_static_real)
        repeated_decoder_hidden = repeat(decoder_hidden)
        if self.encoder.cell_type == "LSTM":
            repeated_prior_states = [repeat(s, dim=1) for s in prior_state]
        else:
            repeated_prior_states = repeat(prior_state, dim=1)

        future_samples = []
        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            current_edge_logits, _, repeated_prior_states, current_inputs = self.unroll(
                prior_state=repeated_prior_states,
                lags=lags,
                scale=repeated_scale,
                time_feat=repeated_time_feat[:, k : k + 1, ...],
                feat_static_cat=repeated_feat_static_cat,
                feat_static_real=repeated_feat_static_real,
                unroll_length=1,
            )

            distr_args, repeated_decoder_hidden, _ = self.decoder(
                current_inputs.squeeze(1),
                hidden=repeated_decoder_hidden,
                edge_logits=current_edge_logits.squeeze(1),
            )

            distr = self.distr_output.distribution(
                distr_args, scale=repeated_scale.squeeze(1)
            )
            new_samples = distr.sample().unsqueeze(1)

            future_samples.append(new_samples)
            repeated_past_target_cdf = torch.cat(
                (repeated_past_target_cdf, new_samples), dim=1
            )

        # (batch_size * num_samples, prediction_length, target_dim)
        samples = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, target_dim)
        return samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length, self.target_dim,)
        )
