import itertools
from typing import Dict, Iterable, Iterator, Optional
import random

import numpy as np
import torch

from pts.transform import Transformation

from .common import DataEntry, Dataset


class TransformedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, dataset: Dataset, is_train: bool, transform: Transformation
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train
        self._cur_iter: Optional[Iterator] = None

    def _iterate_forever(self, collection: Iterable[DataEntry]) -> Iterator[DataEntry]:
        # iterate forever over the collection, the collection must be non empty
        while True:
            try:
                first = next(iter(collection))
            except StopIteration:
                raise Exception("empty dataset")
            else:
                for x in itertools.chain([first], collection):
                    yield x

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        if self._cur_iter is None:
            self._cur_iter = self.transform(
                self._iterate_forever(self.dataset), is_train=self.is_train
            )
        assert self._cur_iter is not None
        while True:
            data_entry = next(self._cur_iter)
            yield {
                k: (v.astype(np.float32) if v.dtype.kind == "f" else v)
                for k, v in data_entry.items()
                if isinstance(v, np.ndarray) == True
            }

    # def __len__(self) -> int:
    #     return len(self.dataset)
