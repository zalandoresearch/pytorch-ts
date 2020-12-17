import time
from typing import List, Optional, Union

from tqdm import tqdm

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader


class Trainer:
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        num_workers: int = 4,
        pin_memory: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def __call__(
        self,
        net: nn.Module,
        train_iter: TrainDataLoader,
        validation_iter: Optional[ValidationDataLoader] = None,
    ) -> None:
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
                    #inputs = [data_entry[k].to(self.device) for k in input_names]

                    output = net(*data_entry.values())
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
                    n_iter = epoch_no * self.num_batches_per_epoch + batch_no
                    #.add_scalar("Loss/train", loss.item(), n_iter)

                    loss.backward()
                    optimizer.step()

                    if self.num_batches_per_epoch == batch_no:
                        # for name, param in net.named_parameters():
                        #     writer.add_histogram(
                        #         name, param.clone().cpu().data.numpy(), n_iter
                        #     )
                        break

            # mark epoch end time and log time cost of current epoch
            toc = time.time()

        #writer.close()
