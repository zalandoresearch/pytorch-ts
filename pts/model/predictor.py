from abc import ABC, abstractmethod
from typing import Iterator, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from pts.dataset import Dataset, DataEntry, InferenceDataLoader
from pts.transform import Transformation

from .forecast import Forecast
from .forecast_generator import ForecastGenerator, SampleForecastGenerator
from .utils import get_module_forward_input_names

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]


class Predictor(ABC):
    def __init__(self, prediction_length: int, freq: str) -> None:
        self.prediction_length = prediction_length
        self.freq = freq

    @abstractmethod
    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        pass


class PTSPredictor(Predictor):
    def __init__(
        self,
        prediction_net: nn.Module,
        batch_size: int,
        prediction_length: int,
        freq: str,
        device: torch.device,
        input_transform: Transformation,
        forecast_generator: ForecastGenerator = SampleForecastGenerator(),
        output_transform: Optional[OutputTransform] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__(prediction_length, freq)

        self.input_names = get_module_forward_input_names(prediction_net)
        self.prediction_net = prediction_net
        self.batch_size = batch_size
        self.input_transform = input_transform
        self.forecast_generator = forecast_generator
        self.output_transform = output_transform
        self.device = device
        self.dtype = dtype

    def predict(self,
                dataset: Dataset,
                num_samples: Optional[int] = None) -> Iterator[Forecast]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            self.input_transform,
            self.batch_size,
            device=self.device,
            dtype=self.dtype,
        )
        with torch.no_grad():
            yield from self.forecast_generator(
                inference_data_loader=inference_data_loader,
                prediction_net=self.prediction_net,
                input_names=self.input_names,
                freq=self.freq,
                output_transform=self.output_transform,
                num_samples=num_samples,
            )
