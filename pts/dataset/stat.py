from collections import defaultdict
from typing import Optional
import math


import numpy as np


class ScaleHistogram:
    """
    Scale histogram of a timeseries dataset
    This counts the number of timeseries whose mean of absolute values is in
    the `[base ** i, base ** (i+1)]` range for all possible `i`.
    The number of entries with empty target is counted separately.
    Parameters
    ----------
    base
        Log-width of the histogram's buckets.
    bin_counts
    empty_target_count
    """

    def __init__(
        self,
        base: float = 2.0,
        bin_counts: Optional[dict] = None,
        empty_target_count: int = 0,
    ) -> None:
        self._base = base
        self.bin_counts = defaultdict(int, {} if bin_counts is None else bin_counts)
        self.empty_target_count = empty_target_count
        self.__init_args__ = dict(
            base=self._base,
            bin_counts=self.bin_counts,
            empty_target_count=empty_target_count,
        )

    def bucket_index(self, target_values):
        assert len(target_values) > 0
        scale = np.mean(np.abs(target_values))
        scale_bin = int(math.log(scale + 1.0, self._base))
        return scale_bin

    def add(self, target_values):
        if len(target_values) > 0:
            bucket = self.bucket_index(target_values)
            self.bin_counts[bucket] = self.bin_counts[bucket] + 1
        else:
            self.empty_target_count = self.empty_target_count + 1

    def count(self, target):
        if len(target) > 0:
            return self.bin_counts[self.bucket_index(target)]
        else:
            return self.empty_target_count

    def __len__(self):
        return self.empty_target_count + sum(self.bin_counts.values())

    def __eq__(self, other):
        return (
            isinstance(other, ScaleHistogram)
            and self.bin_counts == other.bin_counts
            and self.empty_target_count == other.empty_target_count
            and self._base == other._base
        )

    def __str__(self):
        string_repr = [
            "count of scales in {min}-{max}:{count}".format(
                min=self._base ** base_index - 1,
                max=self._base ** (base_index + 1) - 1,
                count=count,
            )
            for base_index, count in sorted(self.bin_counts.items(), key=lambda x: x[0])
        ]
        return "\n".join(string_repr)
