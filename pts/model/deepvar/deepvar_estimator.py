from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pts import Trainer
from pts.model import PTSEstimator, PTSPredictor, copy_parameters
from pts.modules import DistributionOutput, LowRankMultivariateNormalOutput
from pts.dataset import FieldName
from pts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    CDFtoGaussianTransform,
    cdf_to_gaussian_forward_transform,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)
from pts.feature import (
    TimeFeature,
    fourier_time_features_from_frequency_str,
    get_fourier_lags_for_frequency,
)

from .deepvar_network import DeepVARTrainingNetwork, DeepVARPredictionNetwork


class DeepVAREstimator(PTSEstimator):
    def __init__(
        self,
        input_size: int,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "LSTM",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        cardinality: List[int] = [1],
        embedding_dimension: int = 5,
        distr_output: Optional[DistributionOutput] = None,
        rank: Optional[int] = 5,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        conditioning_length: int = 200,
        use_marginal_transformation=False,
        **kwargs,
    ) -> None:
        super().__init__(trainer=trainer, **kwargs)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )

        if distr_output is not None:
            self.distr_output = distr_output
        else:
            self.distr_output = LowRankMultivariateNormalOutput(
                dim=target_dim, rank=rank
            )

        self.input_size = input_size
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.conditioning_length = conditioning_length
        self.use_marginal_transformation = use_marginal_transformation

        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else get_fourier_lags_for_frequency(freq_str=freq)
        )

        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency_str(self.freq)
        )

        self.history_length = self.context_length + max(self.lags_seq)
        self.pick_incomplete = pick_incomplete
        self.scaling = scaling

        if self.use_marginal_transformation:
            self.output_transform: Optional[
                Callable
            ] = cdf_to_gaussian_forward_transform
        else:
            self.output_transform = None

    def create_transformation(self) -> Transformation:
        def use_marginal_transformation(
            marginal_transformation: bool,
        ) -> Transformation:
            if marginal_transformation:
                return CDFtoGaussianTransform(
                    target_field=FieldName.TARGET,
                    observed_values_field=FieldName.OBSERVED_VALUES,
                    max_context_length=self.conditioning_length,
                    target_dim=self.target_dim,
                )
            else:
                return RenameFields(
                    {
                        f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                        f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                    }
                )

        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=1 + len(self.distr_output.event_shape),
                ),
                # maps the target to (1, T)
                # if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=0 if self.distr_output.event_shape[0] == 1 else None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
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
                    pick_incomplete=self.pick_incomplete,
                ),
                use_marginal_transformation(self.use_marginal_transformation),
            ]
        )

    def create_training_network(self, device: torch.device) -> DeepVARTrainingNetwork:
        return DeepVARTrainingNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
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
            conditioning_length=self.conditioning_length,
        ).to(device)

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: DeepVARTrainingNetwork,
        device: torch.device,
    ) -> PTSPredictor:
        prediction_network = DeepVARPredictionNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_parallel_samples=self.num_parallel_samples,
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
            conditioning_length=self.conditioning_length,
        ).to(device)

        copy_parameters(trained_network, prediction_network)

        return PTSPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
            output_transform=self.output_transform,
        )
