from typing import List, Optional

import numpy as np

import torch

from pts import Trainer
from pts.feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    InstanceSplitter,
)
from pts.dataset import FieldName, ExpectedNumInstanceSampler
from pts.model import PTSEstimator
from pts.modules import DistributionOutput, StudentTOutput

from .deepar_network import DeepARTrainingNetwork


class DeepAREstimator(PTSEstimator):
    def __init__(
            self,
            freq: str,
            prediction_length: int,
            trainer: Trainer = Trainer(),
            context_length: Optional[int] = None,
            num_layers: int = 2,
            num_cells: int = 40,
            cell_type: str = "LSTM",
            dropout_rate: float = 0.1,
            use_feat_dynamic_real: bool = False,
            use_feat_static_cat: bool = False,
            use_feat_static_real: bool = False,
            cardinality: Optional[List[int]] = None,
            embedding_dimension: Optional[List[int]] = None,
            distr_output: DistributionOutput = StudentTOutput(),
            scaling: bool = True,
            lags_seq: Optional[List[int]] = None,
            time_features: Optional[List[TimeFeature]] = None,
            num_parallel_samples: int = 100,
            dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__(trainer=trainer)

        self.freq = freq
        self.context_length = (context_length if context_length is not None
                               else prediction_length)
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.distr_output.dtype = dtype
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.dropout_rate = dropout_rate
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat
        self.use_feat_static_real = use_feat_static_real
        self.cardinality = cardinality if cardinality and use_feat_static_cat else [
            1
        ]
        self.embedding_dimension = (
            embedding_dimension if embedding_dimension is not None else
            [min(50, (cat + 1) // 2) for cat in self.cardinality])
        self.scaling = scaling
        self.lags_seq = (lags_seq if lags_seq is not None else
                         get_lags_for_frequency(freq_str=freq))
        self.time_features = (time_features if time_features is not None else
                              time_features_from_frequency_str(self.freq))

        self.history_length = self.context_length + max(self.lags_seq)

        self.num_parallel_samples = num_parallel_samples

    def create_transformation(self) -> Transformation:
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_static_real:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)] +
            ([SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0]
                       )] if not self.use_feat_static_cat else []) +
            ([SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                       )] if not self.use_feat_static_real else []) +
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=np.long,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
                    expected_ndim=1 + len(self.distr_output.event_shape),
                    dtype=self.dtype,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                    dtype=self.dtype,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE] +
                    ([FieldName.FEAT_DYNAMIC_REAL] if self.
                     use_feat_dynamic_real else []),
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.history_length,
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                ),
            ])

    def create_training_network(self, device: torch.device) -> DeepARTrainingNetwork:
        return DeepARTrainingNetwork(
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            dtype=self.dtype).to(device)
