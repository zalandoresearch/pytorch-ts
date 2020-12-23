import time
from typing import List, Optional, Union

from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gluonts.core.component import validated


class Trainer:
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        wandb.init(**kwargs)

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        wandb.watch(net, log="all", log_freq=self.num_batches_per_epoch)

        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        for epoch_no in range(self.epochs):
            # mark epoch start time
            tic = time.time()
            avg_epoch_loss = 0.0

            with tqdm(train_iter) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()
                    inputs = [v.to(self.device) for v in data_entry.values()]

                    output = net(*inputs)
                    if isinstance(output, (list, tuple)):
                        loss = output[0]
                    else:
                        loss = output

                    avg_epoch_loss += loss.item()
                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss": avg_epoch_loss / batch_no,
                            "epoch": epoch_no,
                        },
                        refresh=False,
                    )
                    wandb.log({"loss": loss.item()})

                    loss.backward()
                    optimizer.step()

                    if self.num_batches_per_epoch == batch_no:
                        break

            # mark epoch end time and log time cost of current epoch
            toc = time.time()

        # writer.close()
