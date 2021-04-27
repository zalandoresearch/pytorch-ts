import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import numpy as np
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import json
from gluonts.dataset.common import ListDataset
import torch
from pts.model.deepar import DeepAREstimator
from pts import Trainer
import copy
from datetime import datetime, timedelta


freq = '1D'

prediction_length = 3
context_length = int(prediction_length * 3)

hyperparameters = {
    "time_freq": freq,
    "epochs": "999",
    "early_stopping_patience": "40",
    "mini_batch_size": "128",
    "learning_rate": "5E-4",
    "context_length": str(context_length),
    "prediction_length": str(prediction_length),
    "test_quantiles": [0.16, 0.5, 0.84],
}

TEST_QUANTILES_STR = '["0.16", "0.5", "0.84"]'
LOWER_QT = hyperparameters["test_quantiles"][0]
UPPER_QT = hyperparameters["test_quantiles"][2]

def get_quantile(obs, q):
    return np.quantile(obs, q, axis=0)

def get_upper_lower_qt(forecasts, upper_qt = UPPER_QT, lower_qt = LOWER_QT):
    lower_forecast_qt = np.array(list(map(lambda x: get_quantile(x.samples, lower_qt), forecasts)))
    upper_forecast_qt = np.array(list(map(lambda x: get_quantile(x.samples, upper_qt), forecasts)))
    mean = np.array(list(map(lambda x: get_quantile(x.samples, 0.5), forecasts)))
    return mean, lower_forecast_qt, upper_forecast_qt

def get_time_window_for_forecast_pts(forward_index, json_data, context_length, json_data_train, forecast_len):
    # Censor even number groups
    new_json_data = copy.deepcopy(json_data)
    original_split_date = datetime.strptime(json_data[0]['start'], "%Y-%m-%d")

    if forward_index >= 0:
        for i in range(len(json_data)):
            past_start_date = original_split_date - timedelta(days=(context_length - forward_index))
            new_json_data[i]['target'] = json_data_train[i]['target'][-(context_length - forward_index):] + json_data[i]['target'][:forward_index]
            json_dynam_feat_train = np.array(json_data_train[i]['dynamic_feat'])[:, -(context_length - forward_index):]
            json_dynam_feat_test = np.array(json_data[i]['dynamic_feat'])[:, :forward_index+forecast_len]
            new_json_data[i]['dynamic_feat'] = np.concatenate((json_dynam_feat_train, json_dynam_feat_test), axis=1).tolist()
            new_json_data[i]['start'] = str(past_start_date)
    else:
        for i in range(len(json_data)):            
            past_start_date = original_split_date + timedelta(days=forward_index)
            new_json_data[i]['target'] = json_data_train[i]['target'][forward_index-context_length:forward_index]
            new_json_data[i]['dynamic_feat'] = np.array(json_data_train[i]['dynamic_feat'])[:, forward_index-context_length:forward_index].tolist()
            new_json_data[i]['start'] = str(past_start_date)
    return new_json_data


def rolling_inference_pts(predictor, train_data, test_data, n_forward):
    n_dim = len(train_data)

    mean_ts = np.zeros((n_dim, n_forward))
    lower_forecast_qt_ts = np.zeros((n_dim, n_forward))
    upper_forecast_qt_ts = np.zeros((n_dim, n_forward))

    for f in range(n_forward):
        window_instances = get_time_window_for_forecast_pts(f, test_data, train_data, f)
        window_infernce_data = ListDataset(list(filter(lambda x: {"start": x["start"], "target": x["target"]}, window_instances)), freq = "1D")
        mean, lower_forecast_qt, upper_forecast_qt = get_upper_lower_qt(list(predictor.predict(window_infernce_data)))

        mean_ts[:, f] = mean[:, :1].reshape(-1)# .flatten()
        lower_forecast_qt_ts[:, f] = lower_forecast_qt[:, :1].reshape(-1)# .flatten()
        upper_forecast_qt_ts[:, f] = upper_forecast_qt[:, :1].reshape(-1)# .flatten()

    return mean_ts, lower_forecast_qt_ts, upper_forecast_qt_ts