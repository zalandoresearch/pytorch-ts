from abc import ABC, abstractmethod
from typing import Iterator

from pts.dataset.common import Dataset

from .forecast import Forecast
from .predictor import Predictor


class Predictor(ABC):
    def __init__(self, prediction_length: int, freq: str) -> None:
        self.prediction_length = prediction_length
        self.freq = freq

    @abstractmethod
    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        pass
