from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    Transformation,
    Chain,
    RemoveFields,
    VstackFeatures)

from pts.model.n_beats.base_estimator import BaseNBEATSEstimator
from pts.model.n_beats.n_beats_X_network import NbeatsXTrainingNetwork, NbeatsXPredictionNetwork


class NbeatsXEstimator(BaseNBEATSEstimator):

    training_network_cls = NbeatsXTrainingNetwork
    prediction_network_cls = NbeatsXPredictionNetwork
    valid_n_beats_stack_types = "G",
    time_series_fields = [
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
    ]

    def create_transformation(self) -> Transformation:
        #TODO: add more variable types
        #TODO: Add observed value indicator for other fields? Check how it's usually done
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
