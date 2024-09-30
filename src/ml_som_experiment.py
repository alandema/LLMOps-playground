import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial.distance import pdist
from sklearn import metrics
from minisom import MiniSom
from collections import deque
import mlflow
import os
import json

mlflow.set_experiment('som_experiment')


def filtro_ng(data, labels, tol):
    filtered = deque([data[0]])
    label = deque([labels[0]])
    data_len = len(data)
    for i in range(1, data_len - 1):
        filtered.append(data[i])
        if (pdist(filtered) > tol).all():
            label.append(labels[i])
        else:
            filtered.pop()
    return filtered, label


# trick to overcome complexity. the splits will not be able to calculate each other, thus it is recomended to have the time series ordered and the minimum n_splits as possible
def filtro_ngbatch_list(data, labels, tol, n_splits):
    split_size = len(data) // n_splits
    remainder = len(data) % n_splits
    filtered = []
    filtered_labels = []
    start = 0
    for i in range(n_splits):
        print('Batch:', i, end="\r")
        end = start + split_size + (1 if i < remainder else 0)
        response = filtro_ng(data[start:end], labels[start:end], tol)
        filtered.append(response[0])
        filtered_labels.append(response[1])
        start = end
    return np.hstack([np.concatenate(filtered), np.concatenate(filtered_labels).reshape(-1, 1)])


def train_som(train_data, train_data_labels, parameters):

    with mlflow.start_run():
        mlflow.log_params(parameters)

        scaler = MinMaxScaler()

        train_scaled = scaler.fit_transform(train_data)

        if parameters['do_ng_filtering']:
            treino_filtered = filtro_ngbatch_list(train_scaled,
                                                  train_data_labels.values,
                                                  parameters['tol'],
                                                  parameters['n_batches'])
        else:
            treino_filtered = train_scaled

        som = MiniSom(parameters['nm'],
                      parameters['nm'],
                      treino_filtered.shape[1],
                      sigma=parameters['sigma'],
                      learning_rate=parameters['lr'],
                      neighborhood_function='gaussian',
                      activation_distance='euclidean',
                      random_seed=0)
        som.train(treino_filtered,
                  num_iteration=parameters['num_iteration'],
                  use_epochs=parameters['use_epochs'],
                  verbose=True)

        mlflow.log_metrics({"metric1": 1})


def main(data_folder_path, params_path):
    train_data = pd.read_csv(os.path.join(data_folder_path, 'TRAIN.csv'))
    valid_data = pd.read_csv(os.path.join(data_folder_path, 'VALIDATION.csv'))
    test_data = pd.read_csv(os.path.join(data_folder_path, 'TEST.csv'))

    train_data, train_data_labels = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    valid_data, valid_data_labels = valid_data.iloc[:, :-1], valid_data.iloc[:, -1]
    test_data, test_data_labels = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    with open(params_path) as f:
        parameters = json.load(f)

    train_som(train_data=train_data, train_data_labels=train_data_labels, parameters=parameters)


if __name__ == "__main__":
    data_folder_path = 'data'
    params_path = 'src/params.json'
    main(data_folder_path=data_folder_path, params_path=params_path)
