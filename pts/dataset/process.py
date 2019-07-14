from functools import lru_cache
from typing import Callable, List, cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .common import DataEntry


class ProcessStartField:
    def __init__(self, name: str, freq: str) -> None:
        self.name = name
        self.freq = freq

    def __call__(self, data: DataEntry) -> DataEntry:
        try:
            value = ProcessStartField.process(data[self.name], self.freq)
        except (TypeError, ValueError) as e:
            raise Exception(f'Error "{e}" occurred when reading field "{self.name}"')

        data[self.name] = value

        return data

    @staticmethod
    @lru_cache(maxsize=10000)
    def process(string: str, freq: str) -> pd.Timestamp:
        timestamp = pd.Timestamp(string, freq=freq)
        # 'W-SUN' is the standardized freqstr for W
        if timestamp.freq.name in ("M", "W-SUN"):
            offset = to_offset(freq)
            timestamp = timestamp.replace(
                hour=0, minute=0, second=0, microsecond=0, nanosecond=0
            )
            return pd.Timestamp(offset.rollback(timestamp), freq=offset.freqstr)
        if timestamp.freq == "B":
            # does not floor on business day as it is not allowed
            return timestamp
        return pd.Timestamp(timestamp.floor(timestamp.freq), freq=timestamp.freq)


class ProcessTimeSeriesField:
    def __init__(self, name, is_required: bool, is_static: bool, is_cat: bool) -> None:
        self.name = name
        self.is_required = is_required
        self.req_ndim = 1 if is_static else 2
        self.dtype = np.int32 if is_cat else np.float32

    def __call__(self, data: DataEntry) -> DataEntry:
        value = data.get(self.name, None)

        if value is not None:
            value = np.asarray(value, dtype=self.dtype)
            dim_diff = self.req_ndim - value.ndim
            if dim_diff == 1:
                value = np.expand_dims(a=value, axis=0)
            elif dim_diff != 0:
                raise Exception(
                    f"JSON array has bad shape - expected {self.req_ndim} dimensions got {dim_diff}"
                )

            data[self.name] = value
            return data
        elif not self.is_required:
            return data
        else:
            raise Exception(f"JSON object is missing a required field `{self.name}`")


class ProcessDataEntry:
    def __init__(self, freq: str, one_dim_target: bool = True) -> None:
        self.trans = cast(
            List[Callable[[DataEntry], DataEntry]],
            [
                ProcessStartField("start", freq=freq),
                ProcessTimeSeriesField(
                    "target", is_required=True, is_cat=False, is_static=one_dim_target
                ),
                ProcessTimeSeriesField(
                    "feat_dynamic_cat", is_required=False, is_cat=True, is_static=False
                ),
                ProcessTimeSeriesField(
                    "feat_dynamic_real",
                    is_required=False,
                    is_cat=False,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    "feat_static_cat", is_required=False, is_cat=True, is_static=True
                ),
                ProcessTimeSeriesField(
                    "feat_static_real", is_required=False, is_cat=False, is_static=True
                ),
            ],
        )

    def __call__(self, data: DataEntry) -> DataEntry:
        for t in self.trans:
            data = t(data)
        return data
