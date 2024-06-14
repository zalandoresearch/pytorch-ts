from typing import List, Optional

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddObservedValuesIndicator,
    Transformation,
    Chain,
    RemoveFields,
    AsNumpyArray,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
)

from pts import Trainer

from pts.model.n_beats.base_estimator import BaseNBEATSEstimator
from pts.model.n_beats.n_beats_X_network import (
    NbeatsXTrainingNetwork,
    NbeatsXPredictionNetwork,
)


class NbeatsXEstimator(BaseNBEATSEstimator):

    training_network_cls = NbeatsXTrainingNetwork
    prediction_network_cls = NbeatsXPredictionNetwork
    valid_n_beats_stack_types = ("G",)
    time_series_fields = [
        FieldName.FEAT_TIME,
        FieldName.OBSERVED_VALUES,
    ]

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
        num_stacks: int = 30,
        widths: Optional[List[int]] = None,
        num_blocks: Optional[List[int]] = None,
        num_block_layers: Optional[List[int]] = None,
        expansion_coefficient_lengths: Optional[List[int]] = None,
        sharing: Optional[List[bool]] = None,
        stack_types: Optional[List[str]] = None,
        loss_function: Optional[str] = "MAPE",
        num_feat_dynamic_real: Optional[int] = 0,
        use_feat_dynamic_real: bool = False,
        time_features: Optional[List[TimeFeature]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            num_stacks=num_stacks,
            widths=widths,
            num_blocks=num_blocks,
            num_block_layers=num_block_layers,
            expansion_coefficient_lengths=expansion_coefficient_lengths,
            sharing=sharing,
            stack_types=stack_types,
            loss_function=loss_function,
            num_feat_dynamic_real=num_feat_dynamic_real,
            **kwargs,
        )
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

    def create_transformation(self) -> Transformation:
        remove_field_names = [
            FieldName.FEAT_STATIC_REAL,
            FieldName.FEAT_DYNAMIC_CAT,
            FieldName.FEAT_STATIC_CAT,
        ]
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
                    expected_ndim=1,
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
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
                ),
            ]
        )
