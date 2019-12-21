from typing import Callable, List, NamedTuple, Optional, Tuple, Union

from .common import (
    MetaData,
    CategoricalFeatureInfo,
    BasicFeatureInfo,
    FieldName,
    Dataset,
)
from .list_dataset import ListDataset
from .stat import DatasetStatistics, calculate_dataset_statistics


class DatasetInfo(NamedTuple):
    """
    Information stored on a dataset. When downloading from the repository, the
    dataset repository checks that the obtained version matches the one
    declared in dataset_info/dataset_name.json.
    """

    name: str
    metadata: MetaData
    prediction_length: int
    train_statistics: DatasetStatistics
    test_statistics: DatasetStatistics


def constant_dataset() -> Tuple[DatasetInfo, Dataset, Dataset]:
    metadata = MetaData(
        freq="1H",
        feat_static_cat=[
            CategoricalFeatureInfo(name="feat_static_cat_000", cardinality="10")
        ],
        feat_static_real=[BasicFeatureInfo(name="feat_static_real_000")],
    )

    start_date = "2000-01-01 00:00:00"

    train_ds = ListDataset(
        data_iter=[
            {
                FieldName.ITEM_ID: str(i),
                FieldName.START: start_date,
                FieldName.TARGET: [float(i)] * 24,
                FieldName.FEAT_STATIC_CAT: [i],
                FieldName.FEAT_STATIC_REAL: [float(i)],
            }
            for i in range(10)
        ],
        freq=metadata.freq,
    )

    test_ds = ListDataset(
        data_iter=[
            {
                FieldName.ITEM_ID: str(i),
                FieldName.START: start_date,
                FieldName.TARGET: [float(i)] * 30,
                FieldName.FEAT_STATIC_CAT: [i],
                FieldName.FEAT_STATIC_REAL: [float(i)],
            }
            for i in range(10)
        ],
        freq=metadata.freq,
    )

    info = DatasetInfo(
        name="constant_dataset",
        metadata=metadata,
        prediction_length=6,
        train_statistics=calculate_dataset_statistics(train_ds),
        test_statistics=calculate_dataset_statistics(test_ds),
    )

    return info, train_ds, test_ds
