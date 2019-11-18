from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import torch
from torch.distributions import Distribution

from .quantile import Quantile


class OutputType(str, Enum):
    mean = "mean"
    samples = "samples"
    quantiles = "quantiles"


class Config:
    output_types: Set[OutputType] = {"quantiles", "mean"}
    quantiles: List[str] = ["0.1", "0.5", "0.9"]


class Forecast(ABC):
    start_date: pd.Timestamp
    freq: str
    item_id: Optional[str]
    info: Optional[Dict]
    prediction_length: int
    mean: np.ndarray
    _index = None

    @abstractmethod
    def quantile(self, q: Union[float, str]) -> np.ndarray:
        """
        Computes a quantile from the predicted distribution.

        Parameters
        ----------
        q
            Quantile to compute.

        Returns
        -------
        numpy.ndarray
            Value of the quantile across the prediction range.
        """
        pass

    @abstractmethod
    def dim(self) -> int:
        """
        Returns the dimensionality of the forecast object.
        """
        pass

    @abstractmethod
    def copy_dim(self, dim: int):
        """
        Returns a new Forecast object with only the selected sub-dimension.

        Parameters
        ----------
        dim
            The returned forecast object will only represent this dimension.
        """
        pass

    def as_json_dict(self, config: "Config") -> dict:
        result = {}

        if OutputType.mean in config.output_types:
            result["mean"] = self.mean.tolist()

        if OutputType.quantiles in config.output_types:
            quantiles = map(Quantile.parse, config.quantiles)

            result["quantiles"] = {
                quantile.name: self.quantile(quantile.value).tolist()
                for quantile in quantiles
            }

        if OutputType.samples in config.output_types:
            result["samples"] = []

        return result


class SampleForecast(Forecast):
    """
    A `Forecast` object, where the predicted distribution is represented
    internally as samples.

    Parameters
    ----------
    samples
        Array of size (num_samples, prediction_length)
    start_date
        start of the forecast
    freq
        forecast frequency
    info
        additional information that the forecaster may provide e.g. estimated
        parameters, number of iterations ran etc.
    """

    def __init__(
        self,
        samples: Union[torch.Tensor, np.ndarray],
        start_date,
        freq,
        item_id: Optional[str] = None,
        info: Optional[Dict] = None,
    ):
        assert isinstance(
            samples, (np.ndarray, torch.Tensor)
        ), "samples should be either a numpy array or an torch tensor"
        assert (
            len(np.shape(samples)) == 2 or len(np.shape(samples)) == 3
        ), "samples should be a 2-dimensional or 3-dimensional array. Dimensions found: {}".format(
            len(np.shape(samples))
        )
        self.samples = samples if (isinstance(samples, np.ndarray)) else samples.numpy()
        self._sorted_samples_value = None
        self._mean = None
        self._dim = None
        self.item_id = item_id
        self.info = info

        assert isinstance(
            start_date, pd.Timestamp
        ), "start_date should be a pandas Timestamp object"
        self.start_date = start_date

        assert isinstance(freq, str), "freq should be a string"
        self.freq = freq

    @property
    def _sorted_samples(self):
        if self._sorted_samples_value is None:
            self._sorted_samples_value = np.sort(self.samples, axis=0)
        return self._sorted_samples_value

    @property
    def num_samples(self):
        """
        The number of samples representing the forecast.
        """
        return self.samples.shape[0]

    @property
    def prediction_length(self):
        """
        Time length of the forecast.
        """
        return self.samples.shape[-1]

    @property
    def mean(self):
        """
        Forecast mean.
        """
        if self._mean is not None:
            return self._mean
        else:
            return np.mean(self.samples, axis=0)

    @property
    def mean_ts(self):
        """
        Forecast mean, as a pandas.Series object.
        """
        return pd.Series(self.index, self.mean)

    def quantile(self, q):
        q = Quantile.parse(q).value
        sample_idx = int(np.round((self.num_samples - 1) * q))
        return self._sorted_samples[sample_idx, :]

    def copy_dim(self, dim: int):
        if len(self.samples.shape) == 2:
            samples = self.samples
        else:
            target_dim = self.samples.shape[2]
            assert dim < target_dim, (
                f"must set 0 <= dim < target_dim, but got dim={dim},"
                f" target_dim={target_dim}"
            )
            samples = self.samples[:, :, dim]

        return SampleForecast(
            samples=samples,
            start_date=self.start_date,
            freq=self.freq,
            item_id=self.item_id,
            info=self.info,
        )

    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        else:
            if len(self.samples.shape) == 2:
                # univariate target
                # shape: (num_samples, prediction_length)
                return 1
            else:
                # multivariate target
                # shape: (num_samples, prediction_length, target_dim)
                return self.samples.shape[2]

    def as_json_dict(self, config: "Config") -> dict:
        result = super().as_json_dict(config)

        if OutputType.samples in config.output_types:
            result["samples"] = self.samples.tolist()

        return result

    def __repr__(self):
        return ", ".join(
            [
                f"SampleForecast({self.samples!r})",
                f"{self.start_date!r}",
                f"{self.freq!r}",
                f"item_id={self.item_id!r}",
                f"info={self.info!r})",
            ]
        )


class QuantileForecast(Forecast):
    """
    A Forecast that contains arrays (i.e. time series) for quantiles and mean

    Parameters
    ----------
    forecast_arrays
        An array of forecasts
    start_date
        start of the forecast
    freq
        forecast frequency
    forecast_keys
        A list of quantiles of the form '0.1', '0.9', etc.,
        and potentially 'mean'. Each entry corresponds to one array in
        forecast_arrays.
    info
        additional information that the forecaster may provide e.g. estimated
        parameters, number of iterations ran etc.
    """

    def __init__(
        self,
        forecast_arrays: np.ndarray,
        start_date: pd.Timestamp,
        freq: str,
        forecast_keys: List[str],
        item_id: Optional[str] = None,
        info: Optional[Dict] = None,
    ):
        self.forecast_array = forecast_arrays
        self.start_date = pd.Timestamp(start_date, freq=freq)
        self.freq = freq

        # normalize keys
        self.forecast_keys = [
            Quantile.from_str(key).name if key != "mean" else key
            for key in forecast_keys
        ]
        self.item_id = item_id
        self.info = info
        self._dim = None

        shape = self.forecast_array.shape
        assert shape[0] == len(self.forecast_keys), (
            f"The forecast_array (shape={shape} should have the same "
            f"length as the forecast_keys (len={len(self.forecast_keys)})."
        )
        self.prediction_length = shape[-1]
        self._forecast_dict = {
            k: self.forecast_array[i] for i, k in enumerate(self.forecast_keys)
        }

        self._nan_out = np.array([np.nan] * self.prediction_length)

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        q_str = Quantile.parse(q).name
        # We return nan here such that evaluation runs through
        return self._forecast_dict.get(q_str, self._nan_out)

    @property
    def mean(self):
        """
        Forecast mean.
        """
        return self._forecast_dict.get("mean", self._nan_out)

    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        else:
            if (
                len(self.forecast_array.shape) == 2
            ):  # 1D target. shape: (num_samples, prediction_length)
                return 1
            else:
                return self.forecast_array.shape[
                    1
                ]  # 2D target. shape: (num_samples, target_dim, prediction_length)

    def __repr__(self):
        return ", ".join(
            [
                f"QuantileForecast({self.forecast_array!r})",
                f"start_date={self.start_date!r}",
                f"freq={self.freq!r}",
                f"forecast_keys={self.forecast_keys!r}",
                f"item_id={self.item_id!r}",
                f"info={self.info!r})",
            ]
        )


# class DistributionForecast(Forecast):
#     """
#     A `Forecast` object that uses a distribution directly.
#     This can for instance be used to represent marginal probability
#     distributions for each time point -- although joint distributions are
#     also possible, e.g. when using MultiVariateGaussian).

#     Parameters
#     ----------
#     distribution
#         Distribution object. This should represent the entire prediction
#         length, i.e., if we draw `num_samples` samples from the distribution,
#         the sample shape should be

#            samples = trans_dist.sample(num_samples)
#            samples.shape -> (num_samples, prediction_length)

#     start_date
#         start of the forecast
#     freq
#         forecast frequency
#     info
#         additional information that the forecaster may provide e.g. estimated
#         parameters, number of iterations ran etc.
#     """
#     @validated()
#     def __init__(
#             self,
#             distribution: Distribution,
#             start_date,
#             freq,
#             item_id: Optional[str] = None,
#             info: Optional[Dict] = None,
#     ):
#         self.distribution = distribution
#         self.shape = (self.distribution.batch_shape +
#                       self.distribution.event_shape)
#         self.prediction_length = self.shape[0]
#         self.item_id = item_id
#         self.info = info

#         assert isinstance(
#             start_date,
#             pd.Timestamp), "start_date should be a pandas Timestamp object"
#         self.start_date = start_date

#         assert isinstance(freq, str), "freq should be a string"
#         self.freq = freq
#         self._mean = None

#     @property
#     def mean(self):
#         """
#         Forecast mean.
#         """
#         if self._mean is not None:
#             return self._mean
#         else:
#             self._mean = self.distribution.mean.asnumpy()
#             return self._mean

#     @property
#     def mean_ts(self):
#         """
#         Forecast mean, as a pandas.Series object.
#         """
#         return pd.Series(self.index, self.mean)

#     def quantile(self, level):
#         level = Quantile.parse(level).value
#         q = self.distribution.quantile(mx.nd.array([level])).asnumpy()[0]
#         return q

#     def to_sample_forecast(self, num_samples: int = 200) -> SampleForecast:
#         return SampleForecast(
#             samples=self.distribution.sample(num_samples),
#             start_date=self.start_date,
#             freq=self.freq,
#             item_id=self.item_id,
#             info=self.info,
#         )
