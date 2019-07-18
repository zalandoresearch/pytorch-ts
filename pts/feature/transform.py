from abc import ABC, abstractmethod
from collections import Counter
from functools import lru_cache, reduce
from typing import Iterator, List, Callable, Any, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from pts.dataset import DataEntry, InstanceSampler
from pts import assert_pts
from .time_feature import TimeFeature

MAX_IDLE_TRANSFORMS = 100


@lru_cache(maxsize=10000)
def shift_timestamp(ts: pd.Timestamp, offset: int) -> pd.Timestamp:
    try:
        # this line looks innocent, but can create a date which is out of
        # bounds values over year 9999 raise a ValueError
        # values over 2262-04-11 raise a pandas OutOfBoundsDatetime
        result = ts + offset * ts.freq
        # For freq M and W pandas seems to lose the freq of the timestamp,
        # so we explicitly set it.
        return pd.Timestamp(result, freq=ts.freq)
    except (ValueError, pd._libs.OutOfBoundsDatetime) as ex:
        raise Exception(ex)


def target_transformation_length(
        target: np.array, pred_length: int, is_train: bool
) -> int:
    return target.shape[-1] + (0 if is_train else pred_length)


class Transformation(ABC):
    @abstractmethod
    def __call__(
            self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        pass

    def estimate(self, data_it: Iterator[DataEntry]) -> Iterator[DataEntry]:
        return data_it  # default is to pass through without estimation


class Chain(Transformation):
    """
    Chain multiple transformations together.
    """

    def __init__(self, trans: List[Transformation]) -> None:
        self.trans = trans

    def __call__(
            self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        tmp = data_it
        for t in self.trans:
            tmp = t(tmp, is_train)
        return tmp

    def estimate(self, data_it: Iterator[DataEntry]) -> Iterator[DataEntry]:
        return reduce(lambda x, y: y.estimate(x), self.trans, data_it)


class IdentityTransformation(Transformation):
    def __call__(
            self, data_it: Iterator[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        return data_it


class MapTransformation(Transformation):
    """
    Base class for Transformations that returns exactly one result per input in the stream.
    """

    def __call__(self, data_it: Iterator[DataEntry], is_train: bool) -> Iterator:
        for data_entry in data_it:
            try:
                yield self.map_transform(data_entry.copy(), is_train)
            except Exception as e:
                raise e

    @abstractmethod
    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        pass


class SimpleTransformation(MapTransformation):
    """
    Element wise transformations that are the same in train and test mode
    """

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        return self.transform(data)

    @abstractmethod
    def transform(self, data: DataEntry) -> DataEntry:
        pass


class AdhocTransform(SimpleTransformation):
    """
    Applies a function as a transformation
    This is called ad-hoc, because it is not serializable.
    It is OK to use this for experiments and outside of a model pipeline that
    needs to be serialized.
    """

    def __init__(self, func: Callable[[DataEntry], DataEntry]) -> None:
        self.func = func

    def transform(self, data: DataEntry) -> DataEntry:
        return self.func(data.copy())


class FlatMapTransformation(Transformation):
    """
    Transformations that yield zero or more results per input, but do not combine
    elements from the input stream.
    """

    def __call__(self, data_it: Iterator[DataEntry], is_train: bool) -> Iterator:
        num_idle_transforms = 0
        for data_entry in data_it:
            num_idle_transforms += 1
            try:
                for result in self.flatmap_transform(data_entry.copy(), is_train):
                    num_idle_transforms = 0
                    yield result
            except Exception as e:
                raise e
            if num_idle_transforms > MAX_IDLE_TRANSFORMS:
                raise Exception(
                    f"Reached maximum number of idle transformation calls.\n"
                    f"This means the transformation looped over "
                    f"MAX_IDLE_TRANSFORMS={MAX_IDLE_TRANSFORMS} "
                    f"inputs without returning any output.\n"
                    f"This occurred in the following transformation:\n{self}"
                )

    @abstractmethod
    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        pass


class FilterTransformation(FlatMapTransformation):
    def __init__(self, condition: Callable[[DataEntry], bool]) -> None:
        self.condition = condition

    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        if self.condition(data):
            yield data


class RemoveFields(SimpleTransformation):
    def __init__(self, field_names: List[str]) -> None:
        self.field_names = field_names

    def transform(self, data: DataEntry) -> DataEntry:
        for k in self.field_names:
            if k in data.keys():
                del data[k]
        return data


class SetField(SimpleTransformation):
    """
    Sets a field in the dictionary with the given value.
    Parameters
    ----------
    output_field
        Name of the field that will be set
    value
        Value to be set
    """

    def __init__(self, output_field: str, value: Any) -> None:
        self.output_field = output_field
        self.value = value

    def transform(self, data: DataEntry) -> DataEntry:
        data[self.output_field] = self.value
        return data


class SetFieldIfNotPresent(SimpleTransformation):
    """
    Sets a field in the dictionary with the given value, in case it does not exist already
    Parameters
    ----------
    field
        Name of the field that will be set
    value
        Value to be set
    """

    def __init__(self, field: str, value: Any) -> None:
        self.output_field = field
        self.value = value

    def transform(self, data: DataEntry) -> DataEntry:
        if self.output_field not in data.keys():
            data[self.output_field] = self.value
        return data


class AsNumpyArray(SimpleTransformation):
    """
    Converts the value of a field into a numpy array.
    Parameters
    ----------
    expected_ndim
        Expected number of dimensions. Throws an exception if the number of
        dimensions does not match.
    dtype
        numpy dtype to use.
    """

    def __init__(
            self, field: str, expected_ndim: int, dtype: np.dtype = np.float32
    ) -> None:
        self.field = field
        self.expected_ndim = expected_ndim
        self.dtype = dtype

    def transform(self, data: DataEntry) -> DataEntry:
        value = data[self.field]
        if not isinstance(value, float):
            # this lines produces "ValueError: setting an array element with a
            # sequence" on our test
            # value = np.asarray(value, dtype=np.float32)
            # see https://stackoverflow.com/questions/43863748/
            value = np.asarray(list(value), dtype=self.dtype)
        else:
            # ugly: required as list conversion will fail in the case of a
            # float
            value = np.asarray(value, dtype=self.dtype)
        assert_pts(
            value.ndim >= self.expected_ndim,
            'Input for field "{self.field}" does not have the required'
            "dimension (field: {self.field}, ndim observed: {value.ndim}, "
            "expected ndim: {self.expected_ndim})",
            value=value,
            self=self,
        )
        data[self.field] = value
        return data


class ExpandDimArray(SimpleTransformation):
    """
    Expand dims in the axis specified, if the axis is not present does nothing.
    (This essentially calls np.expand_dims)
    Parameters
    ----------
    field
        Field in dictionary to use
    axis
        Axis to expand (see np.expand_dims for details)
    """

    def __init__(self, field: str, axis: Optional[int] = None) -> None:
        self.field = field
        self.axis = axis

    def transform(self, data: DataEntry) -> DataEntry:
        if self.axis is not None:
            data[self.field] = np.expand_dims(data[self.field], axis=self.axis)
        return data


class VstackFeatures(SimpleTransformation):
    """
    Stack fields together using ``np.vstack``.
    Fields with value ``None`` are ignored.
    Parameters
    ----------
    output_field
        Field name to use for the output
    input_fields
        Fields to stack together
    drop_inputs
        If set to true the input fields will be dropped.
    """

    def __init__(
            self, output_field: str, input_fields: List[str], drop_inputs: bool = True
    ) -> None:
        self.output_field = output_field
        self.input_fields = input_fields
        self.cols_to_drop = (
            []
            if not drop_inputs
            else [fname for fname in self.input_fields if fname != output_field]
        )

    def transform(self, data: DataEntry) -> DataEntry:
        r = [data[fname] for fname in self.input_fields if data[fname] is not None]
        output = np.vstack(r)
        data[self.output_field] = output
        for fname in self.cols_to_drop:
            del data[fname]
        return data


class ConcatFeatures(SimpleTransformation):
    """
    Concatenate fields together using ``np.concatenate``.
    Fields with value ``None`` are ignored.
    Parameters
    ----------
    output_field
        Field name to use for the output
    input_fields
        Fields to stack together
    drop_inputs
        If set to true the input fields will be dropped.
    """

    def __init__(
            self, output_field: str, input_fields: List[str], drop_inputs: bool = True
    ) -> None:
        self.output_field = output_field
        self.input_fields = input_fields
        self.cols_to_drop = (
            []
            if not drop_inputs
            else [fname for fname in self.input_fields if fname != output_field]
        )

    def transform(self, data: DataEntry) -> DataEntry:
        r = [data[fname] for fname in self.input_fields if data[fname] is not None]
        output = np.concatenate(r)
        data[self.output_field] = output
        for fname in self.cols_to_drop:
            del data[fname]
        return data


class SwapAxes(SimpleTransformation):
    """
    Apply `np.swapaxes` to fields.
    Parameters
    ----------
    input_fields
        Field to apply to
    axes
        Axes to use
    """

    def __init__(self, input_fields: List[str], axes: Tuple[int, int]) -> None:
        self.input_fields = input_fields
        self.axis1, self.axis2 = axes

    def transform(self, data: DataEntry) -> DataEntry:
        for field in self.input_fields:
            data[field] = self.swap(data[field])
        return data

    def swap(self, v):
        if isinstance(v, np.ndarray):
            return np.swapaxes(v, self.axis1, self.axis2)
        if isinstance(v, list):
            return [self.swap(x) for x in v]
        else:
            raise ValueError(
                f"Unexpected field type {type(v).__name__}, expected "
                f"np.ndarray or list[np.ndarray]"
            )


class ListFeatures(SimpleTransformation):
    """
    Creates a new field which contains a list of features.
    Parameters
    ----------
    output_field
        Field name for output
    input_fields
        Fields to combine into list
    drop_inputs
        If true the input fields will be removed from the result.
    """

    def __init__(
            self, output_field: str, input_fields: List[str], drop_inputs: bool = True
    ) -> None:
        self.output_field = output_field
        self.input_fields = input_fields
        self.cols_to_drop = (
            []
            if not drop_inputs
            else [fname for fname in self.input_fields if fname != output_field]
        )

    def transform(self, data: DataEntry) -> DataEntry:
        data[self.output_field] = [data[fname] for fname in self.input_fields]
        for fname in self.cols_to_drop:
            del data[fname]
        return data


class AddObservedValuesIndicator(SimpleTransformation):
    """
    Replaces missing values in a numpy array (NaNs) with a dummy value and adds an "observed"-indicator
    that is
      1 - when values are observed
      0 - when values are missing
    Parameters
    ----------
    target_field
        Field for which missing values will be replaced
    output_field
        Field name to use for the indicator
    dummy_value
        Value to use for replacing missing values.
    convert_nans
        If set to true (default) missing values will be replaced. Otherwise
        they will not be replaced. In any case the indicator is included in the
        result.
    """

    def __init__(
            self,
            target_field: str,
            output_field: str,
            dummy_value: int = 0,
            convert_nans: bool = True,
    ) -> None:
        self.dummy_value = dummy_value
        self.target_field = target_field
        self.output_field = output_field
        self.convert_nans = convert_nans

    def transform(self, data: DataEntry) -> DataEntry:
        value = data[self.target_field]
        nan_indices = np.where(np.isnan(value))
        nan_entries = np.isnan(value)

        if self.convert_nans:
            value[nan_indices] = self.dummy_value

        data[self.target_field] = value
        # Invert bool array so that missing values are zeros and store as float
        data[self.output_field] = np.invert(nan_entries).astype(np.float32)
        return data


class RenameFields(SimpleTransformation):
    """
    Rename fields using a mapping
    Parameters
    ----------
    mapping
        Name mapping `input_name -> output_name`
    """

    def __init__(self, mapping: Dict[str, str]) -> None:
        self.mapping = mapping
        values_count = Counter(mapping.values())
        for new_key, count in values_count.items():
            assert count == 1, f"Mapped key {new_key} occurs multiple time"

    def transform(self, data: DataEntry):
        for key, new_key in self.mapping.items():
            if key not in data:
                continue
            assert new_key not in data
            data[new_key] = data[key]
            del data[key]
        return data


class AddConstFeature(MapTransformation):
    """
    Expands a `const` value along the time axis as a dynamic feature, where
    the T-dimension is defined as the sum of the `pred_length` parameter and
    the length of a time series specified by the `target_field`.
    If `is_train=True` the feature matrix has the same length as the `target` field.
    If `is_train=False` the feature matrix has length len(target) + pred_length
    Parameters
    ----------
    output_field
        Field name for output.
    target_field
        Field containing the target array. The length of this array will be used.
    pred_length
        Prediction length (this is necessary since
        features have to be available in the future)
    const
        Constant value to use.
    dtype
        Numpy dtype to use for resulting array.
    """

    def __init__(
            self,
            output_field: str,
            target_field: str,
            pred_length: int,
            const: float = 1.0,
            dtype: np.dtype = np.float32,
    ) -> None:
        self.pred_length = pred_length
        self.const = const
        self.dtype = dtype
        self.output_field = output_field
        self.target_field = target_field

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )
        data[self.output_field] = self.const * np.ones(
            shape=(1, length), dtype=self.dtype
        )
        return data


class AddTimeFeatures(MapTransformation):
    """
    Adds a set of time features.
    If `is_train=True` the feature matrix has the same length as the `target` field.
    If `is_train=False` the feature matrix has length len(target) + pred_length
    Parameters
    ----------
    start_field
        Field with the start time stamp of the time series
    target_field
        Field with the array containing the time series values
    output_field
        Field name for result.
    time_features
        list of time features to use.
    pred_length
        Prediction length
    """

    def __init__(
            self,
            start_field: str,
            target_field: str,
            output_field: str,
            time_features: List[TimeFeature],
            pred_length: int,
    ) -> None:
        self.date_features = time_features
        self.pred_length = pred_length
        self.start_field = start_field
        self.target_field = target_field
        self.output_field = output_field
        self._min_time_point: Optional[pd.Timestamp] = None
        self._max_time_point: Optional[pd.Timestamp] = None
        self._full_range_date_features: Optional[np.ndarray] = None
        self._date_index: Optional[pd.DatetimeIndex] = None

    def _update_cache(self, start: pd.Timestamp, length: int) -> None:
        end = shift_timestamp(start, length)
        if self._min_time_point is not None:
            if self._min_time_point <= start and end <= self._max_time_point:
                return
        if self._min_time_point is None:
            self._min_time_point = start
            self._max_time_point = end
        self._min_time_point = min(shift_timestamp(start, -50), self._min_time_point)
        self._max_time_point = max(shift_timestamp(end, 50), self._max_time_point)
        self.full_date_range = pd.date_range(
            self._min_time_point, self._max_time_point, freq=start.freq
        )
        self._full_range_date_features = np.vstack(
            [feat(self.full_date_range) for feat in self.date_features]
        )
        self._date_index = pd.Series(
            index=self.full_date_range, data=np.arange(len(self.full_date_range))
        )

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        start = data[self.start_field]
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )
        self._update_cache(start, length)
        i0 = self._date_index[start]
        features = self._full_range_date_features[..., i0:i0 + length]
        data[self.output_field] = features
        return data


class AddAgeFeature(MapTransformation):
    """
    Adds an 'age' feature to the data_entry.
    The age feature starts with a small value at the start of the time series
    and grows over time.
    If `is_train=True` the age feature has the same length as the `target` field.
    If `is_train=False` the age feature has length len(target) + pred_length
    Parameters
    ----------
    target_field
        Field with target values (array) of time series
    output_field
        Field name to use for the output.
    pred_length
        Prediction length
    log_scale
        If set to true the age feature grows logarithmically otherwise linearly over time.
    """

    def __init__(
            self,
            target_field: str,
            output_field: str,
            pred_length: int,
            log_scale: bool = True,
    ) -> None:
        self.pred_length = pred_length
        self.target_field = target_field
        self.feature_name = output_field
        self.log_scale = log_scale
        self._age_feature = np.zeros(0)

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )

        if self.log_scale:
            age = np.log10(2.0 + np.arange(length, dtype=np.float32))
        else:
            age = np.arange(length, dtype=np.float32)

        data[self.feature_name] = age.reshape((1, length))

        return data


class InstanceSplitter(FlatMapTransformation):
    """
    Selects training instances, by slicing the target and other time series
    like arrays at random points in training mode or at the last time point in
    prediction mode. Assumption is that all time like arrays start at the same
    time point.
    The target and each time_series_field is removed and instead two
    corresponding fields with prefix `past_` and `future_` are included. E.g.
    If the target array is one-dimensional, the resulting instance has shape
    (len_target). In the multi-dimensional case, the instance has shape (dim,
    len_target).
    target -> past_target and future_target
    The transformation also adds a field 'past_is_pad' that indicates whether
    values where padded or not.
    Convention: time axis is always the last axis.
    Parameters
    ----------
    target_field
        field containing the target
    is_pad_field
        output field indicating whether padding happened
    start_field
        field containing the start date of the time series
    forecast_start_field
        output field that will contain the time point where the forecast starts
    train_sampler
        instance sampler that provides sampling indices given a time-series
    past_length
        length of the target seen before making prediction
    future_length
        length of the target that must be predicted
    batch_first
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout
    time_series_fields
        fields that contains time-series, they are split in the same interval
        as the target
    pick_incomplete
        whether training examples can be sampled with only a part of
        past_length time-units
        present for the time series. This is useful to train models for
        cold-start. In such case, is_pad_out contains an indicator whether
        data is padded or not.
    """

    def __init__(
            self,
            target_field: str,
            is_pad_field: str,
            start_field: str,
            forecast_start_field: str,
            train_sampler: InstanceSampler,
            past_length: int,
            future_length: int,
            batch_first: bool = True,
            time_series_fields: Optional[List[str]] = None,
            pick_incomplete: bool = True,
    ) -> None:

        assert future_length > 0

        self.train_sampler = train_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.batch_first = batch_first
        self.ts_fields = time_series_fields if time_series_fields is not None else []
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.pick_incomplete = pick_incomplete

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        pl = self.future_length
        slice_cols = self.ts_fields + [self.target_field]
        target = data[self.target_field]

        len_target = target.shape[-1]

        if is_train:
            if len_target < self.future_length:
                # We currently cannot handle time series that are shorter than
                # the prediction length during training, so we just skip these.
                # If we want to include them we would need to pad and to mask
                # the loss.
                sampling_indices: List[int] = []
            else:
                if self.pick_incomplete:
                    sampling_indices = self.train_sampler(
                        target, 0, len_target - self.future_length
                    )
                else:
                    sampling_indices = self.train_sampler(
                        target, self.past_length, len_target - self.future_length
                    )
        else:
            sampling_indices = [len_target]
        for i in sampling_indices:
            pad_length = max(self.past_length - i, 0)
            if not self.pick_incomplete:
                assert pad_length == 0
            d = data.copy()
            for ts_field in slice_cols:
                if i > self.past_length:
                    # truncate to past_length
                    past_piece = d[ts_field][..., i - self.past_length: i]
                elif i < self.past_length:
                    pad_block = np.zeros(
                        d[ts_field].shape[:-1] + (pad_length,), dtype=d[ts_field].dtype
                    )
                    past_piece = np.concatenate(
                        [pad_block, d[ts_field][..., :i]], axis=-1
                    )
                else:
                    past_piece = d[ts_field][..., :i]
                d[self._past(ts_field)] = past_piece
                d[self._future(ts_field)] = d[ts_field][..., i: i + pl]
                del d[ts_field]
            pad_indicator = np.zeros(self.past_length)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1

            if self.batch_first:
                for ts_field in slice_cols:
                    d[self._past(ts_field)] = d[self._past(ts_field)].transpose()
                    d[self._future(ts_field)] = d[self._future(ts_field)].transpose()

            d[self._past(self.is_pad_field)] = pad_indicator
            d[self.forecast_start_field] = shift_timestamp(d[self.start_field], i)
            yield d


class CanonicalInstanceSplitter(FlatMapTransformation):
    """
    Selects instances, by slicing the target and other time series
    like arrays at random points in training mode or at the last time point in
    prediction mode. Assumption is that all time like arrays start at the same
    time point.
    In training mode, the returned instances contain past_`target_field`
    as well as past_`time_series_fields`.
    In prediction mode, one can set `use_prediction_features` to get
    future_`time_series_fields`.
    If the target array is one-dimensional, the `target_field` in the resulting instance has shape
    (`instance_length`). In the multi-dimensional case, the instance has shape (`dim`, `instance_length`),
    where `dim` can also take a value of 1.
    In the case of insufficient number of time series values, the
    transformation also adds a field 'past_is_pad' that indicates whether
    values where padded or not, and the value is padded with
    `default_pad_value` with a default value 0.
    This is done only if `allow_target_padding` is `True`,
    and the length of `target` is smaller than `instance_length`.
    Parameters
    ----------
    target_field
        fields that contains time-series
    is_pad_field
        output field indicating whether padding happened
    start_field
        field containing the start date of the time series
    forecast_start_field
        field containing the forecast start date
    instance_sampler
        instance sampler that provides sampling indices given a time-series
    instance_length
        length of the target seen before making prediction
    batch_first
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout
    time_series_fields
        fields that contains time-series, they are split in the same interval
        as the target
    allow_target_padding
        flag to allow padding
    pad_value
        value to be used for padding
    use_prediction_features
        flag to indicate if prediction range features should be returned
    prediction_length
        length of the prediction range, must be set if
        use_prediction_features is True
    """

    def __init__(
            self,
            target_field: str,
            is_pad_field: str,
            start_field: str,
            forecast_start_field: str,
            instance_sampler: InstanceSampler,
            instance_length: int,
            batch_first: bool = True,
            time_series_fields: List[str] = [],
            allow_target_padding: bool = False,
            pad_value: float = 0.0,
            use_prediction_features: bool = False,
            prediction_length: Optional[int] = None,
    ) -> None:
        self.instance_sampler = instance_sampler
        self.instance_length = instance_length
        self.batch_first = batch_first
        self.dynamic_feature_fields = time_series_fields
        self.target_field = target_field
        self.allow_target_padding = allow_target_padding
        self.pad_value = pad_value
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field

        assert (
                not use_prediction_features or prediction_length is not None
        ), "You must specify `prediction_length` if `use_prediction_features`"

        self.use_prediction_features = use_prediction_features
        self.prediction_length = prediction_length

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        ts_fields = self.dynamic_feature_fields + [self.target_field]
        ts_target = data[self.target_field]

        len_target = ts_target.shape[-1]

        if is_train:
            if len_target < self.instance_length:
                sampling_indices = (
                    # Returning [] for all time series will cause this to be in loop forever!
                    [len_target]
                    if self.allow_target_padding
                    else []
                )
            else:
                sampling_indices = self.instance_sampler(
                    ts_target, self.instance_length, len_target
                )
        else:
            sampling_indices = [len_target]

        for i in sampling_indices:
            d = data.copy()

            pad_length = max(self.instance_length - i, 0)

            # update start field
            d[self.start_field] = shift_timestamp(
                data[self.start_field], i - self.instance_length
            )

            # set is_pad field
            is_pad = np.zeros(self.instance_length)
            if pad_length > 0:
                is_pad[:pad_length] = 1
            d[self.is_pad_field] = is_pad

            # update time series fields
            for ts_field in ts_fields:
                full_ts = data[ts_field]
                if pad_length > 0:
                    pad_pre = self.pad_value * np.ones(
                        shape=full_ts.shape[:-1] + (pad_length,)
                    )
                    past_ts = np.concatenate([pad_pre, full_ts[..., :i]], axis=-1)
                else:
                    past_ts = full_ts[..., (i - self.instance_length): i]

                past_ts = past_ts.transpose() if self.batch_first else past_ts
                d[self._past(ts_field)] = past_ts

                if self.use_prediction_features and not is_train:
                    if not ts_field == self.target_field:
                        future_ts = full_ts[..., i: i + self.prediction_length]
                        future_ts = (
                            future_ts.transpose() if self.batch_first else future_ts
                        )
                        d[self._future(ts_field)] = future_ts

                del d[ts_field]

            d[self.forecast_start_field] = shift_timestamp(
                d[self.start_field], self.instance_length
            )

            yield d


class SelectFields(MapTransformation):
    """
    Only keep the listed fields
    Parameters
    ----------
    input_fields
        List of fields to keep.
    """

    def __init__(self, input_fields: List[str]) -> None:
        self.input_fields = input_fields

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        return {f: data[f] for f in self.input_fields}
