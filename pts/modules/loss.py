import torch
import torch.nn as nn

from .soft_dtw import SoftDTWBatch, pairwise_distances
from .path_soft_dtw import PathDTWBatch


def smape_loss(forecast: torch.Tensor, future_target: torch.Tensor) -> torch.Tensor:
    denominator = (torch.abs(future_target) + torch.abs(forecast)).detach()
    flag = denominator == 0

    return (200 / self.prediction_length) * torch.mean(
        (torch.abs(future_target - forecast) * torch.logical_not(flag))
        / (denominator + flag),
        dim=1,
    )


def mape_loss(forecast: torch.Tensor, future_target: torch.Tensor) -> torch.Tensor:
    denominator = torch.abs(future_target)
    flag = denominator == 0

    return (100 / self.prediction_length) * torch.mean(
        (torch.abs(future_target - forecast) * torch.logical_not(flag))
        / (denominator + flag),
        dim=1,
    )


def mase_loss(
    forecast: torch.Tensor,
    future_target: torch.Tensor,
    past_target: torch.Tensor,
    periodicity: int,
) -> torch.Tensor:
    factor = 1 / (self.context_length + self.prediction_length - periodicity)

    whole_target = torch.cat((past_target, future_target), dim=1)
    seasonal_error = factor * torch.mean(
        torch.abs(
            whole_target[:, periodicity:, ...] - whole_target[:, :-periodicity:, ...]
        ),
        dim=1,
    )
    flag = seasonal_error == 0

    return (
        torch.mean(torch.abs(future_target - forecast), dim=1) * torch.logical_not(flag)
    ) / (seasonal_error + flag)


def dilate_loss(forecast, future_target, alpha=0.5, gamma=0.01):
    batch_size, N_output = forecast.shape

    pairwise_distance = torch.zeros((batch_size, N_output, N_output)).to(
        forecast.device
    )
    for k in range(batch_size):
        Dk = pairwise_distances(
            future_target[k, :].view(-1, 1), forecast[k, :].view(-1, 1)
        )
        pairwise_distance[k : k + 1, :, :] = Dk

    softdtw_batch = SoftDTWBatch.apply
    loss_shape = softdtw_batch(pairwise_distance, gamma)

    path_dtw = PathDTWBatch.apply
    path = path_dtw(pairwise_distance, gamma)

    omega = pairwise_distances(torch.arange(1, N_output + 1).view(N_output, 1)).to(
        forecast.device
    )
    loss_temporal = torch.sum(path * omega) / (N_output * N_output)

    return alpha * loss_shape + (1 - alpha) * loss_temporal
