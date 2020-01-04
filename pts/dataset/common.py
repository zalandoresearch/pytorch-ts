from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, NamedTuple, Sized, List, Optional, Iterator

import pandas as pd
from pydantic import BaseModel

DataEntry = Dict[str, Any]


class SourceContext(NamedTuple):
    source: str
    row: int


class FieldName:
    """
    A bundle of default field names to be used by clients when instantiating
    transformer instances.
    """

    ITEM_ID = "item_id"

    START = "start"
    TARGET = "target"

    FEAT_STATIC_CAT = "feat_static_cat"
    FEAT_STATIC_REAL = "feat_static_real"
    FEAT_DYNAMIC_CAT = "feat_dynamic_cat"
    FEAT_DYNAMIC_REAL = "feat_dynamic_real"

    FEAT_TIME = "time_feat"
    FEAT_CONST = "feat_dynamic_const"
    FEAT_AGE = "feat_dynamic_age"

    OBSERVED_VALUES = "observed_values"
    IS_PAD = "is_pad"
    FORECAST_START = "forecast_start"


class Dataset(Sized, Iterable[DataEntry], ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[DataEntry]:
        pass

    @abstractmethod
    def __len__(self):
        pass


class CategoricalFeatureInfo(BaseModel):
    name: str
    cardinality: str


class BasicFeatureInfo(BaseModel):
    name: str


class MetaData(BaseModel):
    freq: str = None
    target: Optional[BasicFeatureInfo] = None

    feat_static_cat: List[CategoricalFeatureInfo] = []
    feat_static_real: List[BasicFeatureInfo] = []
    feat_dynamic_real: List[BasicFeatureInfo] = []
    feat_dynamic_cat: List[CategoricalFeatureInfo] = []

    prediction_length: Optional[int] = None


class TrainDatasets(NamedTuple):
    """
    A dataset containing two subsets, one to be used for training purposes,
    and the other for testing purposes, as well as metadata.
    """

    metadata: MetaData
    train: Dataset
    test: Optional[Dataset] = None

class DateConstants:
    """
    Default constants for specific dates.
    """

    OLDEST_SUPPORTED_TIMESTAMP = pd.Timestamp(1800, 1, 1, 12)
    LATEST_SUPPORTED_TIMESTAMP = pd.Timestamp(2200, 1, 1, 12)
