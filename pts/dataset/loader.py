from typing import Callable, Iterable, Iterator, List, Optional
import itertools

from torch.utils.data import IterableDataset

from gluonts.dataset.common import Dataset
from gluonts.transform import Transformation, TransformedDataset
from gluonts.itertools import cyclic, pseudo_shuffled


class TransformedIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        is_train: bool = True,
        shuffle_buffer_length: Optional[int] = None,
    ):
        super().__init__()
        self.shuffle_buffer_length = shuffle_buffer_length

        self.transformed_dataset = TransformedDataset(
            cyclic(dataset),
            transform,
            is_train=is_train,
        )

    def __iter__(self):
        if self.shuffle_buffer_length is None:
            return iter(self.transformed_dataset)
        else:
            return pseudo_shuffled(
                iter(self.transformed_dataset),
                shuffle_buffer_length=self.shuffle_buffer_length,
            )
