from functools import partial

from gluonts.dataset.repository.datasets import get_download_path, dataset_recipes

from ._m5 import generate_pts_m5_dataset

dataset_recipes["pts_m5"] = partial(
    generate_pts_m5_dataset,
    pandas_freq="D",
    prediction_length=28,
    m5_file_path=get_download_path() / "m5",
)
