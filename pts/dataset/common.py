from abc import ABC, abstractmethod

from typing import Any, Dict, Sized, Iterable, NamedTuple

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
    def __iter__(self) -> Iterable[DataEntry]:
        pass

    @abstractmethod
    def __len__(self):
        pass
