from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import (
    Normal,
    Uniform,
    Bernoulli)
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.torch.modules.distribution_output import DistributionOutput
from pts import Trainer
from pts.model.deepar import DeepAREstimator
from pts.model.simple_feedforward import SimpleFeedForwardEstimator
from pts.modules import (
    ImplicitQuantileOutput
)

NUM_SAMPLES = 2000
BATCH_SIZE = 32
TOL = 0.3
START_TOL_MULTIPLE = 1


def inv_softplus(y: np.ndarray) -> np.ndarray:
    return np.log(np.exp(y) - 1)


def learn_distribution(
    distr_output: DistributionOutput,
    samples: torch.Tensor,
    init_biases: List[np.ndarray] = None,
    num_epochs: int = 5,
    learning_rate: float = 1e-2,
):
    arg_proj = distr_output.get_args_proj(in_features=1)

    if init_biases is not None:
        for param, bias in zip(arg_proj.proj, init_biases):
            nn.init.constant_(param.bias, bias)

    dummy_data = torch.ones((len(samples), 1, 1))

    dataset = TensorDataset(dummy_data, samples)
    train_data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = SGD(arg_proj.parameters(), lr=learning_rate)

    for e in range(num_epochs):
        cumulative_loss = 0
        num_batches = 0

        for i, (data, sample_label) in enumerate(train_data):
            optimizer.zero_grad()
            distr_args = arg_proj(data)
            distr = distr_output.distribution(distr_args)
            loss = -distr.log_prob(sample_label).mean()
            loss.backward()
            clip_grad_norm_(arg_proj.parameters(), 10.0)
            optimizer.step()

            num_batches += 1
            cumulative_loss += loss.item()
        print("Epoch %s, loss: %s" % (e, cumulative_loss / num_batches))

    sampling_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    i, (data, sample_label) = next(enumerate(sampling_dataloader))
    distr_args = arg_proj(data)
    distr = distr_output.distribution(distr_args)
    samples = distr.sample((NUM_SAMPLES, ))

    with torch.no_grad():
        percentile_90 = distr.quantile_function(torch.ones((1, 1, 1)), torch.ones((1, 1)) * 0.9)
        percentile_10 = distr.quantile_function(torch.ones((1, 1, 1)), torch.ones((1, 1)) * 0.1)

    return samples.mean(), samples.std(), percentile_10, percentile_90


def test_independent_implicit_quantile() -> None:
    num_samples = NUM_SAMPLES

    # # Normal distrib
    distr_mean = torch.Tensor([10.])
    distr_std = torch.Tensor([4.])
    distr_pp10 = distr_mean - 1.282 * distr_std
    distr_pp90 = distr_mean + 1.282 * distr_std
    distr = Normal(loc=distr_mean, scale=distr_std)

    samples = distr.sample((num_samples,))
    learned_mean, learned_std, learned_pp10, learned_pp90 = learn_distribution(
        ImplicitQuantileOutput(output_domain="Real"),
        samples=samples,
        num_epochs=50,
        learning_rate=1e-2
    )

    torch.testing.assert_allclose(learned_mean, distr_mean.squeeze(), rtol=0.1, atol=0.1*10)
    torch.testing.assert_allclose(learned_std, distr_std.squeeze(), rtol=0.1, atol=.1*4)
    torch.testing.assert_allclose(learned_pp90, distr_pp90.squeeze(), rtol=0.1, atol=.1 * 4)
    torch.testing.assert_allclose(learned_pp10, distr_pp10.squeeze(), rtol=0.1, atol=.1 * 4)

    # Uniform distrib
    a = torch.Tensor([0.])
    b = torch.Tensor([20.])
    distr_mean = 0.5*(a+b)
    distr_std = (1./12.*(b-a)**2)**0.5
    distr_pp10 = 0.1 * (a+b)
    distr_pp90 = 0.9 * (a+b)
    distr = Uniform(low=a, high=b)

    samples = distr.sample((num_samples,))
    learned_mean, learned_std, learned_pp10, learned_pp90 = learn_distribution(
        ImplicitQuantileOutput(output_domain="Positive"),
        samples=samples,
        num_epochs=50,
        learning_rate=1e-2
    )

    torch.testing.assert_allclose(learned_mean, distr_mean.squeeze(), atol=1., rtol=0.1)
    torch.testing.assert_allclose(learned_std, distr_std.squeeze(), atol=0.5, rtol=0.1)
    torch.testing.assert_allclose(learned_pp90, distr_pp90.squeeze(), rtol=0.1, atol=.1 * 18)
    torch.testing.assert_allclose(learned_pp10, distr_pp10.squeeze(), rtol=0.2, atol=.2 * 2)

    # Bernoulli distrib
    distr_mean = torch.Tensor([0.2])
    distr_std = distr_mean * (1 - distr_mean)
    distr_pp10 = torch.Tensor([0.])
    distr_pp90 = torch.Tensor([1.])
    distr = Bernoulli(probs=distr_mean)

    samples = distr.sample((num_samples,))
    learned_mean, learned_std, learned_pp10, learned_pp90 = learn_distribution(
        ImplicitQuantileOutput(output_domain="Positive"),
        samples=samples,
        num_epochs=50,
        learning_rate=1e-2
    )

    torch.testing.assert_allclose(learned_mean, distr_mean.squeeze(), atol=1., rtol=0.1)
    torch.testing.assert_allclose(learned_std, distr_std.squeeze(), atol=0.5, rtol=0.1)
    torch.testing.assert_allclose(learned_pp90, distr_pp90.squeeze(), rtol=0.1, atol=.1 * 18)
    torch.testing.assert_allclose(learned_pp10, distr_pp10.squeeze(), rtol=0.1, atol=.1 * 2)


def test_training_with_implicit_quantile_output():
    dataset = get_dataset("constant")
    metadata = dataset.metadata

    deepar_estimator = DeepAREstimator(
        distr_output=ImplicitQuantileOutput(output_domain="Real"),
        freq=metadata.freq,
        prediction_length=metadata.prediction_length,
        trainer=Trainer(device="cpu",
                        epochs=5,
                        learning_rate=1e-3,
                        num_batches_per_epoch=3,
                        batch_size=256,
                        num_workers=1,
                        ),
        input_size=48,
    )
    deepar_predictor = deepar_estimator.train(dataset.train)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,  # test dataset
        predictor=deepar_predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator(num_workers=0)
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset.test))

    assert agg_metrics["MSE"] > 0


def test_instanciation_of_args_proj():

    class MockedImplicitQuantileOutput(ImplicitQuantileOutput):
        method_calls = 0

        @classmethod
        def set_args_proj(cls):
            super().set_args_proj()
            cls.method_calls += 1

    dataset = get_dataset("constant")
    metadata = dataset.metadata

    distr_output = MockedImplicitQuantileOutput(output_domain="Real")

    deepar_estimator = DeepAREstimator(
        distr_output=distr_output,
        freq=metadata.freq,
        prediction_length=metadata.prediction_length,
        trainer=Trainer(device="cpu",
                        epochs=1,
                        learning_rate=1e-3,
                        num_batches_per_epoch=1,
                        batch_size=256,
                        num_workers=1,
                        ),
        input_size=48,
    )
    assert distr_output.method_calls == 1
    deepar_predictor = deepar_estimator.train(dataset.train)

    # Method should be called when the MockedImplicitQuantileOutput is instanciated,
    # and one more time because in_features is different from 1
    assert distr_output.method_calls == 2

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,  # test dataset
        predictor=deepar_predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator(num_workers=0)
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset.test))
    assert distr_output.method_calls == 2

    # Test that the implicit output module is proper reset
    new_estimator = DeepAREstimator(
        distr_output=MockedImplicitQuantileOutput(output_domain="Real"),
        freq=metadata.freq,
        prediction_length=metadata.prediction_length,
        trainer=Trainer(device="cpu",
                        epochs=1,
                        learning_rate=1e-3,
                        num_batches_per_epoch=1,
                        batch_size=256,
                        num_workers=1,
                        ),
        input_size=48,
    )
    assert distr_output.method_calls == 3
    new_estimator.train(dataset.train)
    assert distr_output.method_calls == 3  # Since in_feature is the same as before, there should be no additional call
