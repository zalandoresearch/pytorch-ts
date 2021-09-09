from typing import List, Optional

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    Transformation,
    Chain,
    RemoveFields,
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

    def create_transformation(self) -> Transformation:
        # TODO: add more variable types
        # TODO: Add observed value indicator for other fields? Check how it's usually done
        return Chain(
            [
                RemoveFields(
                    field_names=[
                        FieldName.FEAT_STATIC_REAL,
                        # FieldName.FEAT_DYNAMIC_REAL,
                        FieldName.FEAT_DYNAMIC_CAT,
                    ]
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_DYNAMIC_REAL],
                ),
            ]
        )
