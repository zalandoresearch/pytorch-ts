# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonts.torch.util import weighted_average

from .module import DNRIModel


class DNRILightningModule(pl.LightningModule):
    def __init__(
        self,
        model: DNRIModel,
        kl_coef: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.kl_coef = kl_coef
        self.lr = lr
        self.weight_decay = weight_decay

    def _compute_loss(self, batch):
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]

        # encoder
        (
            prior_logits,
            posterior_logits,
            _,
            scale,
            _,
            encoder_input,
        ) = self.model.unroll_encoder(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        # decoder
        all_distr_args = []
        num_time_steps = encoder_input.size(1)
        decoder_hidden = self.decoder.get_initial_hidden(
            encoder_input.size(), device=encoder_input.device
        )
        for step in range(num_time_steps):
            current_inputs = encoder_input[:, step]
            current_p_logits = posterior_logits[:, step]

            distr_args, decoder_hidden, _ = self.decoder(
                inputs=current_inputs,
                hidden=decoder_hidden,
                edge_logits=current_p_logits,
            )
            all_distr_args.append(distr_args)

        map_stack = partial(torch.stack, dim=1)
        all_distr_args = tuple(map(map_stack, zip(*all_distr_args)))
        distr = self.model.output_distribution(all_distr_args, scale=scale)

        context_target = past_target[:, -self.model.context_length + 1 :]
        target = torch.cat(
            (context_target, future_target),
            dim=1,
        )

        loss_nll = -distr.log_prob(target).unsqueeze(-1)
        prob = F.softmax(posterior_logits, dim=-1)
        loss_kl = self.kl_categorical_learned(prob, prior_logits)

        loss_values = loss_nll + self.kl_coef * loss_kl

        context_observed = past_observed_values[:, -self.model.context_length + 1 :]
        observed_values = torch.cat((context_observed, future_observed_values), dim=1)

        if len(self.model.target_shape) == 0:
            loss_weights = observed_values
        else:
            loss_weights = observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss_values, weights=loss_weights)

    @staticmethod
    def kl_categorical_learned(preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds * (torch.log(preds + 1e-16) - log_prior)

        return kl_div.view(preds.size(0), preds.size(1), -1).sum(dim=-1, keepdims=True)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        train_loss = self._compute_loss(batch)
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss = self._compute_loss(batch)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
