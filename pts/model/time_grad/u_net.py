import math

import torch
from torch import nn
import torch.nn.functional as F


# @torch.jit.script
# def mish(input):
#     """
#     Applies the mish function element-wise:
#     mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
#     """
#     return input * torch.tanh(F.softplus(input))


# class Mish(nn.Module):
#     def forward(self, x):
#         return mish(x)


# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim, linear_scale=5000):
#         super().__init__()
#         self.dim = dim
#         self.linear_scale = linear_scale

#     def forward(self, noise_level):
#         half_dim = self.dim // 2
#         exponents = torch.arange(half_dim, dtype=torch.float32).to(noise_level) / float(
#             half_dim
#         )
#         exponents = 1e-4 ** exponents
#         exponents = (
#             self.linear_scale * noise_level.unsqueeze(-1) * exponents.unsqueeze(0)
#         )
#         return torch.cat((exponents.sin(), exponents.cos()), dim=-1)


# class UNet(nn.Module):
#     def __init__(self, dim, cond_dim, time_emb_dim=4):
#         super().__init__()

#         self.time_pos_emb = SinusoidalPosEmb(time_emb_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(time_emb_dim, dim // 2),
#             Mish(),
#             nn.Linear(dim // 2, dim),
#         )

#         self.downs = nn.Sequential(
#             Mish(),
#             nn.Linear(dim, dim),
#             Mish(),
#             nn.Linear(dim, dim//2),
#             Mish(),
#             nn.Linear(dim//2, dim//2),
#         )

#         self.ups = nn.Sequential(
#             Mish(),
#             nn.Linear(dim//2 + cond_dim, dim // 2),
#             Mish(),
#             nn.Linear(dim // 2, dim//2),
#             Mish(),
#             nn.Linear(dim // 2, dim),
#             Mish(),
#             nn.Linear(dim, dim),
#             Mish(),
#             nn.Linear(dim, dim),
#             Mish(),
#         )

#     def forward(self, x, time, cond):
#         t = self.time_pos_emb(time)
#         t = self.mlp(t)

#         h = self.downs(x + t)

#         h = torch.cat((h, cond), -1)

#         return self.ups(h)


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = x * torch.sigmoid(x)
        x = self.projection2(x)
        x = x * torch.sigmoid(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(1, 2 * residual_channels, 1, padding=2)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


# class CondUpsampler(nn.Module):
#     def __init__(self, kernel_size=4, stride=2, padding=1):
#         super().__init__()
#         self.conv1 = nn.Conv1d(
#             1, 1, kernel_size, stride=stride, padding=padding
#         )
#         self.conv2 = nn.Conv1d(
#             1, 1, kernel_size, stride=stride, padding=padding
#         )

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.leaky_relu(x, 0.4)
#         x = self.conv2(x)
#         x = F.leaky_relu(x, 0.4)
#         return x


class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear = nn.Linear(cond_length, target_dim)

    def forward(self, x):
        x = F.leaky_relu(x, 0.4)
        x = self.linear(x)
        x = F.leaky_relu(x, 0.4)
        return x


class TimeDiff(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_length,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(1, residual_channels, 1, padding=2)
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=target_dim, cond_length=cond_length
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):
        x = self.input_projection(inputs)
        x = F.elu(x)

        diffusion_step = self.diffusion_embedding(time)
        cond_up = self.cond_upsampler(cond)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x
