from typing import Union, List

import numpy as np

from abc import ABC, abstractmethod

from .stat import ScaleHistogram

class InstanceSampler(ABC):
    @abstractmethod
    def __call__(self, ts: np.ndarray, a: int, b: int) -> Union[np.ndarray, List[int]]:
        pass


class UniformSplitSampler(InstanceSampler):
    """
    Samples each point with the same fixed probability.
    Parameters
    ----------
    p
        Probability of selecting a time point
    """

    def __init__(self, p: float = 1.0 / 20.0) -> None:
        self.p = p
        self.lookup = np.arange(2 ** 13)

    def __call__(self, ts: np.ndarray, a: int, b: int) -> np.ndarray:
        assert a <= b
        while ts.shape[-1] >= len(self.lookup):
            self.lookup = np.arange(2 * len(self.lookup))
        mask = np.random.uniform(low=0.0, high=1.0, size=b - a + 1) < self.p
        return self.lookup[a : a + len(mask)][mask]


class TestSplitSampler(InstanceSampler):
    """
    Sampler used for prediction. Always selects the last time point for
    splitting i.e. the forecast point for the time series.
    """

    def __call__(self, ts: np.ndarray, a: int, b: int) -> np.ndarray:
        return np.array([b])


class ExpectedNumInstanceSampler(InstanceSampler):
    """
    Keeps track of the average time series length and adjusts the probability
    per time point such that on average `num_instances` training examples are
    generated per time series.
    Parameters
    ----------
    num_instances
        number of training examples generated per time series on average
    """

    def __init__(self, num_instances: float) -> None:
        self.num_instances = num_instances
        self.avg_length = 0.0
        self.n = 0.0
        self.lookup = np.arange(2 ** 13)

    def __call__(self, ts: np.ndarray, a: int, b: int) -> np.ndarray:
        while ts.shape[-1] >= len(self.lookup):
            self.lookup = np.arange(2 * len(self.lookup))

        self.n += 1.0
        self.avg_length += float(b - a + 1 - self.avg_length) / float(self.n)
        p = self.num_instances / self.avg_length

        mask = np.random.uniform(low=0.0, high=1.0, size=b - a + 1) < p
        indices = self.lookup[a : a + len(mask)][mask]
        return indices


class BucketInstanceSampler(InstanceSampler):
    """
    This sample can be used when working with a set of time series that have a
    skewed distributions. For instance, if the dataset contains many time series
    with small values and few with large values.
    The probability of sampling from bucket i is the inverse of its number of elements.
    Parameters
    ----------
    scale_histogram
        The histogram of scale for the time series. Here scale is the mean abs
        value of the time series.
    """

    def __init__(self, scale_histogram: ScaleHistogram) -> None:
        # probability of sampling a bucket i is the inverse of its number of
        # elements
        self.scale_histogram = scale_histogram
        self.lookup = np.arange(2 ** 13)

    def __call__(self, ts: np.ndarray, a: int, b: int) -> np.ndarray:
        while ts.shape[-1] >= len(self.lookup):
            self.lookup = np.arange(2 * len(self.lookup))
        p = 1.0 / self.scale_histogram.count(ts)
        mask = np.random.uniform(low=0.0, high=1.0, size=b - a + 1) < p
        indices = self.lookup[a : a + len(mask)][mask]
        return indices
