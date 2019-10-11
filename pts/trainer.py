import time
from typing import Any, List, NamedTuple, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from .dataset import TrainDataLoader


class Trainer:
    def __init__(self,
                 epochs: int = 100,
                 batch_size: int = 32,
                 num_batches_per_epoch: int = 50,
                 learning_rate: float = 1e-3,
                 device: Optional[torch.device] = None) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.device = device

    def __call__(self, net: nn.Module, input_names: List[str],
                 train_iter: TrainDataLoader) -> None:

        net.to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)

        for epoch_no in range(self.epochs):
            # mark epoch start time
            tic = time.time()

            with tqdm(train_iter) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()
                    inputs = [data_entry[k] for k in input_names]

                    output = net(*inputs)
                    if isinstance(output, (list, tuple)):
                        loss = output[0]
                    else:
                        loss = output

                    loss.backward()
                    optimizer.step()
            
            # mark epoch end time and log time cost of current epoch
            toc = time.time()