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

import boto3
from io import BytesIO
import pickle as pkl

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
    # "test_quantiles": [0.16, 0.5, 0.84]
    "test_quantiles": [0.1, 0.5, 0.9],
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

def get_time_window_for_forecast_pts(forward_index, json_data, json_data_train, context_length, prediction_length):
    # Censor even number groups
    new_json_data = copy.deepcopy(json_data)
    original_split_date = datetime.strptime(json_data[0]['start'], "%Y-%m-%d")

    if forward_index >= 0:
        for i in range(len(json_data)):
            past_start_date = original_split_date - timedelta(days=(context_length - forward_index))
            data_train_sample = json_data_train[i]['target'][-(context_length - forward_index):]
            data_test_samnple = json_data[i]['target'][:forward_index ]

            new_json_data[i]['target'] = np.concatenate([ json_data_train[i]['target'][-(context_length - forward_index):] , json_data[i]['target'][:forward_index]])
            json_dynam_feat_train = np.array(json_data_train[i]['feat_dynamic_real'])[:, -(context_length - forward_index):]
            json_dynam_feat_test = np.array(json_data[i]['feat_dynamic_real'])[:, :forward_index + prediction_length]
            new_json_data[i]['feat_dynamic_real'] = np.concatenate((json_dynam_feat_train, json_dynam_feat_test), axis=1).tolist()

            json_control_train = np.array(json_data_train[i]['control'])[ -(context_length - forward_index):]
            json_control_test = np.array(json_data[i]['control'])[ :forward_index + prediction_length]
            new_json_data[i]['control'] = np.concatenate((json_control_train, json_control_test)).tolist()


            new_json_data[i]['start'] = str(past_start_date)
    else:
        for i in range(len(json_data)):            
            past_start_date = original_split_date + timedelta(days=forward_index)
            new_json_data[i]['target'] = json_data_train[i]['target'][forward_index-context_length:forward_index]
            new_json_data[i]['feat_dynamic_real'] = np.array(json_data_train[i]['feat_dynamic_real'])[:, forward_index-context_length:forward_index + prediction_length].tolist()

            new_json_data[i]['control'] = np.array(json_data_train[i]['control'])[ forward_index-context_length:forward_index+forecast_len + prediction_length].tolist()

            new_json_data[i]['start'] = str(past_start_date)
    return new_json_data


def rolling_inference_pts(predictor, train_data, test_data, n_forward):
    n_dim = len(train_data)

    mean_ts = np.zeros((n_dim, n_forward))
    lower_forecast_qt_ts = np.zeros((n_dim, n_forward))
    upper_forecast_qt_ts = np.zeros((n_dim, n_forward))

    for f in range(n_forward):
        window_instances = get_time_window_for_forecast_pts(f, test_data, train_data, context_length, prediction_length)
        window_infernce_data = ListDataset(window_instances, freq = "1D")
        # window_infernce_data = ListDataset(list(filter(lambda x: {"start": x["start"], "target": x["target"]}, window_instances)), freq = "1D")
        mean, lower_forecast_qt, upper_forecast_qt = get_upper_lower_qt(list(predictor.predict(window_infernce_data)))

        mean_ts[:, f] = mean[:, :1].reshape(-1)# .flatten()
        lower_forecast_qt_ts[:, f] = lower_forecast_qt[:, :1].reshape(-1)# .flatten()
        upper_forecast_qt_ts[:, f] = upper_forecast_qt[:, :1].reshape(-1)# .flatten()

    return mean_ts, lower_forecast_qt_ts, upper_forecast_qt_ts



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    special_id = "split-id-376582_de_2019-01-07_ooh_placeholder_ONOFF-dates-2019-01-07_2020-01-08_2020-01-22"
    predictor = torch.load(f"inference_output/predictor_obj_{special_id}_epoch_12.pt", map_location=torch.device('cpu'))
    predictor.device = 'cpu'
    
    bucket = 'ab-testing-dpa'
    training_filename = f"pytorch-ts-format/pytorch_ts-training_data-{special_id}.pkl"
    test_filename = f"pytorch-ts-format/pytorch_ts-test_data-{special_id}.pkl"

    s3 = boto3.resource('s3')
    with BytesIO() as data:
        s3.Bucket(bucket).download_fileobj(training_filename, data)
        data.seek(0) 
        training_data = pkl.load(data)

    with BytesIO() as data:
        s3.Bucket(bucket).download_fileobj(test_filename, data)
        data.seek(0) 
        test_data = pkl.load(data)
    
    forward_index = 0
    time_window = get_time_window_for_forecast_pts(forward_index, test_data, training_data, context_length, prediction_length)
    
    # time_window_input = ListDataset(time_window, freq = "1D")
    time_window_input = ListDataset(time_window, freq = "1D")

    sample_prediction = list(predictor.predict(time_window_input))

    n_forward = 11
    mean_ts, lower_forecast_qt_ts, upper_forecast_qt_ts = rolling_inference_pts(predictor, training_data, test_data, n_forward)

    print("testing done")

