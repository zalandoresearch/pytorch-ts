import numpy as np
import pytest

from gluonts.dataset.artificial import recipe as rcp
from gluonts.dataset.common import ListDataset, MetaData
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from pts.model.n_beats import NBEATSEstimator
from pts import Trainer
from pts.model.n_beats.n_beats_X_estimator import NbeatsXEstimator

NUM_EPOCHS = 1
NUM_BATCHES_PER_EPOCH = 16
NUM_SAMPLES = 1 # It's a pointwise estimate, don't need more...
BATCH_SIZE = 32

TIME_SERIE_LENGTH = 4 * 12
PREDICTION_LENGTH = 4
NUMBER_OF_TIME_SERIES = NUM_BATCHES_PER_EPOCH * BATCH_SIZE

META_DATA = MetaData(freq="W", prediction_length=PREDICTION_LENGTH)


class SimpleDiscount(rcp.Lifted):
    def __call__(self, x, length, *args, **kwargs):
        return np.random.uniform(0, 0.7, size=length)


class SimpleTarget(rcp.Lifted):

    def __call__(self, x, length, *args, **kwargs):
        trend = np.arange(length)
        season = np.asarray([np.cos(2 * np.pi * i * 1/4) for i in range(length)])
        return 0.2 * trend + .3 * season + 0.3 * x


@pytest.fixture
def dataset():
    list_for_dataset = []
    for _ in range(NUMBER_OF_TIME_SERIES):
        discounts = SimpleDiscount()(x=None, length=TIME_SERIE_LENGTH)
        sales = SimpleTarget()(x=discounts, length=TIME_SERIE_LENGTH)

        list_for_dataset.append({
            "feat_dynamic_real": [discounts],
            "target": sales,
            "start": "2019-01-07 00:00"}
        )

    return ListDataset(list_for_dataset, freq=META_DATA.freq)


# TODO: do the same for configurable NBEATS
@pytest.fixture
def estimator():
    return NbeatsXEstimator(
        freq=META_DATA.freq,
        prediction_length=META_DATA.prediction_length,
        context_length=META_DATA.prediction_length * 3,
        trainer=Trainer(device="cpu",
                        epochs=NUM_EPOCHS,
                        learning_rate=1e-3,
                        num_batches_per_epoch=NUM_BATCHES_PER_EPOCH,
                        batch_size=BATCH_SIZE,
                        ),
        num_stacks=30,
        num_blocks=[1],
        num_block_layers=[4],
        stack_types=["G"],
        widths=[512],
        sharing=[False],
        expansion_coefficient_lengths=[32]
    )


def train_and_evaluate(estimator, dataset):
    predictor = estimator.train(dataset)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,  # test dataset
        predictor=predictor,  # predictor
        num_samples=NUM_SAMPLES,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator()
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset))

    return agg_metrics, item_metrics


def test_nbeats_convergence(estimator, dataset):
    agg_metrics, item_metrics = train_and_evaluate(estimator, dataset)
    print(agg_metrics["NRMSE"])
    assert agg_metrics["NRMSE"] < 1.  # After one epoch, should be at least below 100% error...
