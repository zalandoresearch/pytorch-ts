from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn

from pts.dataset import InferenceDataLoader, DataEntry
from pts.model import Forecast, DistributionForecast
from pts.modules import DistributionOutput

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]


class ForecastGenerator(ABC):
    """
    Classes used to bring the output of a network into a class.
    """
    @abstractmethod
    def __call__(self, 
                 inference_data_loader: InferenceDataLoader,
                 prediction_net: nn.Module,
                 input_names: List[str],
                 freq: str,
                 output_transform: Optional[OutputTransform],
                 num_samples: Optional[int], 
                 **kwargs) -> Iterator[Forecast]:
        pass


class DistributionForecastGenerator(ForecastGenerator):
    def __init__(self, distr_output: DistributionOutput) -> None:
        self.distr_output = distr_output

    def __call__(self,
                 inference_data_loader: InferenceDataLoader,
                 prediction_net: nn.Module,
                 input_names: List[str],
                 freq: str,
                 output_transform: Optional[OutputTransform],
                 num_samples: Optional[int], 
                 **kwargs) -> Iterator[DistributionForecast]:
        