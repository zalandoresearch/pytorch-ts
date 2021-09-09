from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    Transformation,
    Chain,
    RemoveFields,
)

from pts.model.n_beats.base_estimator import BaseNBEATSEstimator

from .n_beats_network import (
    NBEATSPredictionNetwork,
    NBEATSTrainingNetwork,
    VALID_N_BEATS_STACK_TYPES,
)


class NBEATSEstimator(BaseNBEATSEstimator):

    training_network_cls = NBEATSTrainingNetwork
    prediction_network_cls = NBEATSPredictionNetwork
    valid_n_beats_stack_types = VALID_N_BEATS_STACK_TYPES
    time_serie_fields = [FieldName.OBSERVED_VALUES]

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                RemoveFields(
                    field_names=[
                        FieldName.FEAT_STATIC_REAL,
                        FieldName.FEAT_DYNAMIC_REAL,
                        FieldName.FEAT_DYNAMIC_CAT,
                    ]
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                ),
            ]
        )
