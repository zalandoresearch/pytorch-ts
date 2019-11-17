from typing import Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Scaler(ABC, nn.Module):
    def __init__(self, keepdim: bool = False):
        super().__init__()
        self.keepdim = keepdim

    @abstractmethod
    def compute_scale(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:
        pass

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        data
            tensor of shape (N, T, C) containing the data to be scaled

        observed_indicator
            observed_indicator: binary tensor with the same shape as
            ``data``, that has 1 in correspondence of observed data points,
            and 0 in correspondence of missing data points.

        Returns
        -------
        Tensor
            Tensor containing the "scaled" data, shape: (N, T, C).
        Tensor
            Tensor containing the scale, of shape (N, C) if ``keepdim == False``, and shape
            (N, 1, C) if ``keepdim == True``.
        """

        scale = self.compute_scale(data, observed_indicator)

        if self.keepdim:
            scale = scale.unsqueeze(1)
            return data / scale, scale
        else:
            return data / scale.unsqueeze(1), scale


class MeanScaler(Scaler):
    """
    The ``MeanScaler`` computes a per-item scale according to the average
    absolute value over time of each item. The average is computed only among
    the observed values in the data tensor, as indicated by the second
    argument. Items with no observed data are assigned a scale based on the
    global average.

    Parameters
    ----------
    minimum_scale
        default scale that is used if the time series has only zeros.
    """

    def __init__(self, minimum_scale: float = 1e-10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.minimum_scale = minimum_scale

    def compute_scale(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:
        # these will have shape (N, C)
        num_observed = observed_indicator.sum(dim=1)
        sum_observed = (data.abs() * observed_indicator).sum(dim=1)

        # first compute a global scale per-dimension
        total_observed = num_observed.sum(dim=0)
        denominator = torch.max(total_observed, torch.tensor(1.0))
        default_scale = sum_observed.sum(dim=0) / denominator

        # then compute a per-item, per-dimension scale
        denominator = torch.max(num_observed, torch.tensor(1.0))
        scale = sum_observed / denominator

        # use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        scale = torch.where(
            sum_observed > torch.zeros_like(sum_observed),
            scale,
            default_scale * torch.ones_like(num_observed),
        )

        return torch.max(scale, torch.tensor(self.minimum_scale))


class NOPScaler(Scaler):
    """
    The ``NOPScaler`` assigns a scale equals to 1 to each input item, i.e.,
    no scaling is applied upon calling the ``NOPScaler``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_scale(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:
        return torch.ones_like(data).mean(dim=1)

