import math

import torch
from torch import nn
import torch.nn.functional as F


@torch.jit.script
def mish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    def forward(self, x):
        return mish(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        #emb = x[:, None] * emb[None, :]
        emb = x.unsqueeze(-1) * emb
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class UNet(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear((dim//2)*2, dim*4), 
            Mish(), 
            nn.Linear(dim*4, dim)
        )

        self.downs = nn.Sequential(
            Mish(),
            nn.Linear(dim, dim//4),
            Mish(),
            nn.Linear(dim//4, dim//4),
        )

        self.ups = nn.Sequential(
            Mish(),
            nn.Linear(dim//4 + cond_dim, dim//4),
            Mish(),
            nn.Linear(dim//4, dim),
            Mish(),
        )

    def forward(self, x, time, cond):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = self.downs(x + t)

        h = torch.cat((h, cond), -1)

        return self.ups(h)



# class UNet(nn.Module):
#     def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), groups=8):
#         super().__init__()
#         dims = [3, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))

#         self.time_pos_emb = SinusoidalPosEmb(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
#         )

#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)

#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)

#             self.downs.append(
#                 nn.ModuleList(
#                     [
#                         ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
#                         ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
#                         Residual(Rezero(LinearAttention(dim_out))),
#                         Downsample(dim_out) if not is_last else nn.Identity(),
#                     ]
#                 )
#             )

#         mid_dim = dims[-1]
#         self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
#         self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
#         self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
#             is_last = ind >= (num_resolutions - 1)

#             self.ups.append(
#                 nn.ModuleList(
#                     [
#                         ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
#                         ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
#                         Residual(Rezero(LinearAttention(dim_in))),
#                         Upsample(dim_in) if not is_last else nn.Identity(),
#                     ]
#                 )
#             )

#         out_dim = default(out_dim, 3)
#         self.final_conv = nn.Sequential(Block(dim, dim), nn.Conv2d(dim, out_dim, 1))

#     def forward(self, x, time, hidden=None):
#         t = self.time_pos_emb(time)
#         t = self.mlp(t)

#         h = []

#         for resnet, resnet2, attn, downsample in self.downs:
#             x = resnet(x, t)
#             x = resnet2(x, t)
#             x = attn(x)
#             h.append(x)
#             x = downsample(x)

#         x = self.mid_block1(x, t)
#         x = self.mid_attn(x)
#         x = self.mid_block2(x, t)

#         for resnet, resnet2, attn, upsample in self.ups:
#             x = torch.cat((x, h.pop()), dim=1)
#             x = resnet(x, t)
#             x = resnet2(x, t)
#             x = attn(x)
#             x = upsample(x)

#         return self.final_conv(x)