import time
from typing import List, Optional, Union

from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
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
        maximum_learning_rate: float = 1e-2,
        wandb_mode: str = "disabled",
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        self.device = device
        wandb.init(mode=wandb_mode, **kwargs)

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        wandb.watch(net, log="all", log_freq=self.num_batches_per_epoch)

        optimizer = Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )

        for epoch_no in range(self.epochs):
            # mark epoch start time
            tic = time.time()
            avg_epoch_loss = 0.0

            if validation_iter is not None:
                it_val = tqdm(validation_iter)
                val_iter_obj = list(zip(it_val, range(1, validation_iter.batch_size+1)))
                train_iter_obj = enumerate(tqdm(train_iter), start=1)

            with tqdm(train_iter) as it:
                
                for batch_no, data_entry in enumerate(it, start=1):

                    optimizer.zero_grad()

                    # Strong assumption that validation_iter and train_iter are same iter size
                    if validation_iter is not None:
                        val_batch = val_iter_obj[batch_no-1][1]
                        inputs_val = [v.to(self.device) for v in data_entry.values()]
                        output_val = net(*inputs_val)

                        if isinstance(output_val, (list, tuple)):
                            loss_val = output_val[0]
                        else:
                            loss_val = output_val
                    
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
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)

                    optimizer.step()
                    lr_scheduler.step()

                    if self.num_batches_per_epoch == batch_no:
                        break
                
                    print("validation loss: ")
                    print(loss_val.item())

            # mark epoch end time and log time cost of current epoch
            toc = time.time()

        # writer.close()
