from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pts.dataset import Dataset, TransformedIterableDataset
from pts.transform import Transformation
from pts import Trainer

from .predictor import Predictor
from .utils import get_module_forward_input_names


class Estimator(ABC):
    prediction_length: int
    freq: str

    @abstractmethod
    def train(self, training_data: Dataset) -> Predictor:
        pass


class DummyEstimator(Estimator):
    """
    An `Estimator` that, upon training, simply returns a pre-constructed
    `Predictor`.

    Parameters
    ----------
    predictor_cls
        `Predictor` class to instantiate.
    **kwargs
        Keyword arguments to pass to the predictor constructor.
    """

    def __init__(self, predictor_cls: type, **kwargs) -> None:
        self.predictor = predictor_cls(**kwargs)

    def train(self, training_data: Dataset) -> Predictor:
        return self.predictor


class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: nn.Module
    predictor: Predictor


class PTSEstimator(Estimator):
    def __init__(self, trainer: Trainer, dtype: np.dtype = np.float32) -> None:
        self.trainer = trainer
        self.dtype = dtype

    @abstractmethod
    def create_transformation(self) -> Transformation:
        """
        Create and return the transformation needed for training and inference.

        Returns
        -------
        Transformation
            The transformation that will be applied entry-wise to datasets,
            at training and inference time.
        """
        pass

    @abstractmethod
    def create_training_network(self, device: torch.device) -> nn.Module:
        """
        Create and return the network used for training (i.e., computing the
        loss).

        Returns
        -------
        nn.Module
            The network that computes the loss given input data.
        """
        pass

    @abstractmethod
    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
    ) -> Predictor:
        """
        Create and return a predictor object.

        Returns
        -------
        Predictor
            A predictor wrapping a `nn.Module` used for inference.
        """
        pass

    def train_model(self, training_data: Dataset) -> TrainOutput:
        transformation = self.create_transformation()
        transformation.estimate(iter(training_data))

        training_iter_dataset = TransformedIterableDataset(
            dataset=training_data,
            is_train=True,
            transform=transformation
        )

        training_data_loader = DataLoader(
            training_iter_dataset,
            batch_size=self.trainer.batch_size,
            num_workers=self.train.num_workers,
            pip_memory=self.trainer.pip_memory
        )

        # ensure that the training network is created on the same device
        trained_net = self.create_training_network(self.trainer.device)

        self.trainer(
            net=trained_net,
            input_names=get_module_forward_input_names(trained_net),
            data_loader=training_data_loader,
        )

        return TrainOutput(
            transformation=transformation,
            trained_net=trained_net,
            predictor=self.create_predictor(
                transformation, trained_net, self.trainer.device
            ),
        )

    def train(self, training_data: Dataset) -> Predictor:
        return self.train_model(training_data).predictor
