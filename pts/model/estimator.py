from abc import ABC, abstractmethod

import numpy as np

from pts.dataset.common import Dataset
from pts.dataset import TrainDataLoader
from pts.feature import Transformation

import torch
import torch.nn as nn

from .predictor import Predictor


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
    def __init__(self, trainer: Trainer,
                 float_type: np.dtype = np.float32) -> None:
        self.trainer = trainer
        self.float_type = float_type
        

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
        HybridBlock
            The network that computes the loss given input data.
        """
        pass

    @abstractmethod
    def create_predictor(self, transformation: Transformation,
                         trained_network: nn.Module) -> Predictor:
        """
        Create and return a predictor object.

        Returns
        -------
        Predictor
            A predictor wrapping a `HybridBlock` used for inference.
        """
        pass

    def train_model(self, training_data: Dataset) -> TrainOutput:
        transformation = self.create_transformation()

        transformation.estimate(iter(training_data))

        training_data_loader = TrainDataLoader(
            dataset=training_data,
            transform=transformation,
            batch_size=self.trainer.batch_size,
            num_batches_per_epoch=self.trainer.num_batches_per_epoch,
            device=self.trainer.device,
            float_type=self.float_type,
        )

        # ensure that the training network is created on the same device
        trained_net = self.create_training_network(self.device)

        self.trainer(
            net=trained_net,
            input_names=get_hybrid_forward_input_names(trained_net),
            train_iter=training_data_loader,
        )

        with self.trainer.ctx:
            # ensure that the prediction network is created within the same MXNet
            # context as the one that was used during training
            return TrainOutput(
                transformation=transformation,
                trained_net=trained_net,
                predictor=self.create_predictor(transformation, trained_net),
            )

    def train(self, training_data: Dataset) -> Predictor:
        return self.train_model(training_data).predictor
