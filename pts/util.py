from typing import List

import torch

from gluonts.torch.util import slice_along_dim


def lagged_sequence_values(
    indices: List[int],
    prior_sequence: torch.Tensor,
    sequence: torch.Tensor,
    dim: int = 1,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Constructs an array of lagged values from a given sequence.

    Parameters
    ----------
    indices
        Indices of the lagged observations. For example, ``[0]`` indicates
        that, at any time ``t``, the will have only the observation from
        time ``t`` itself; instead, ``[0, 24]`` indicates that the output
        will have observations from times ``t`` and ``t-24``.
    prior_sequence
        Tensor containing the input sequence prior to the time range for
        which the output is required.
    sequence
        Tensor containing the input sequence in the time range where the
        output is required.
    dim
        Time dimension.
    keepdim
        Whether to keep the last dimension of the output tensor.

    Returns
    -------
    Tensor
        A tensor of shape (*sequence.shape, len(indices)).
    """
    assert max(indices) <= prior_sequence.shape[dim], (
        f"lags cannot go further than prior sequence length, found lag"
        f" {max(indices)} while prior sequence is only"
        f"{prior_sequence.shape[dim]}-long"
    )

    # if prior_sequence is a 2-tensor add an extra dimension
    if len(prior_sequence.shape) == 2:
        prior_sequence = prior_sequence.unsqueeze(-1)
    if len(sequence.shape) == 2:
        sequence = sequence.unsqueeze(-1)

    full_sequence = torch.cat((prior_sequence, sequence), dim=dim)

    lags_values = []
    for lag_index in indices:
        begin_index = -lag_index - sequence.shape[dim]
        end_index = -lag_index if lag_index > 0 else None
        lags_values.append(
            slice_along_dim(
                full_sequence, dim=dim, slice_=slice(begin_index, end_index)
            ).unsqueeze(-1)
        )

    lags_values = torch.cat(lags_values, dim=-1)

    if not keepdim:
        # merge the last two dimensions
        lags_values = lags_values.reshape(*lags_values.shape[:-2], -1)

    return lags_values
