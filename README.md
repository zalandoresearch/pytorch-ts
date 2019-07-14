# PyTorch-TS

PyTorch-TS is a PyTorch Probabilistic Time Series Modeling framework which provides state of the art time series models and utilities for loading and iterating over time series datasets.

## Installation

```
$ pip3 install pytorch-ts
```

## Usage

```python
import pandas as pd

from pts.dataset import ListDataset

url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AAPL.csv"
df = pd.read_csv(url, header=0, index_col=0)

training_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-05 00:00:00"]}],
    freq = "5min"
)
```
