from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic
from gluonts.dataset.loader import as_stacked_batches
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.util import copy_parameters
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)
from gluonts.transform.sampler import InstanceSampler

from .lightning_module import LSTNetLightningModule

PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values"]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class LSTNetEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        prediction_length: Optional[int],
        context_length: int,
        input_size: int,
        ar_window: int = 24,
        skip_size: int = 24,
        channels: int = 96,
        kernel_size: int = 6,
        horizon: Optional[int] = None,
        dropout_rate: Optional[float] = 0.1,
        output_activation: Optional[str] = None,
        rnn_cell_type: str = "GRU",
        rnn_num_cells: int = 100,
        skip_rnn_cell_type: str = "GRU",
        skip_rnn_num_cells: int = 5,
        scaling: Optional[str] = "mean",
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        loss=nn.L1Loss(reduce=False),
        lr: float = 1e-3,
        patience: int = 10,
        weight_decay: float = 1e-8,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ):
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.input_size = input_size
        self.skip_size = skip_size
        self.ar_window = ar_window
        self.horizon = horizon
        self.prediction_length = prediction_length

        self.future_length = horizon if horizon is not None else prediction_length
        self.context_length = context_length
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.rnn_cell_type = rnn_cell_type
        self.rnn_num_cells = rnn_num_cells
        self.skip_rnn_cell_type = skip_rnn_cell_type
        self.skip_rnn_num_cells = skip_rnn_num_cells
        self.scaling = scaling
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.num_parallel_samples = num_parallel_samples
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=2),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
            ]
        )

    def _create_instance_splitter(self, module: LSTNetLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
            output_NTC=True,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: LSTNetLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: LSTNetLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_lightning_module(self) -> LSTNetLightningModule:
        return LSTNetLightningModule(
            loss=self.loss,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            model_kwargs=dict(
                input_size=self.input_size,
                channels=self.channels,
                kernel_size=self.kernel_size,
                rnn_cell_type=self.rnn_cell_type,
                rnn_num_cells=self.rnn_num_cells,
                skip_rnn_cell_type=self.skip_rnn_cell_type,
                skip_rnn_num_cells=self.skip_rnn_num_cells,
                skip_size=self.skip_size,
                ar_window=self.ar_window,
                context_length=self.context_length,
                horizon=self.horizon,
                prediction_length=self.prediction_length,
                dropout_rate=self.dropout_rate,
                output_activation=self.output_activation,
                scaling=self.scaling,
            ),
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: LSTNetLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
