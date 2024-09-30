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


def generate_train_map_figure(train_filtered, train_label, som):
    # plotting the distance map as background
    w_y = []
    for xx in train_filtered[train_label == 1]:
        w_y.append(som.winner(xx))  # getting the winner
    w_y = np.array(w_y)
    w_y = np.unique(w_y, axis=0)

    w_n = []
    for xx in train_filtered[train_label == 0]:
        w_n.append(som.winner(xx))  # getting the winner
    w_n = np.array(w_n)
    w_n = np.unique(w_n, axis=0)

    fig = plt.figure(figsize=(6, 6))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    if len(w_y) > 0:
        plt.scatter(w_y[:, 0] + .5, w_y[:, 1] + .5, color='red', marker='s', s=50, label='RCG', alpha=0.2)
    if len(w_n) > 0:
        plt.scatter(w_n[:, 0] + .5, w_n[:, 1] + .5, color='blue', marker='o', s=30, label='RSG', alpha=0.2)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.tight_layout()

    return fig


def generate_test_map_figure(test_data, test_data_labels, som):
    # plotting the distance map as background
    w_y = []
    for xx in test_data[test_data_labels == 1]:
        w_y.append(som.winner(xx))  # getting the winner
    w_y = np.array(w_y)
    w_y = np.unique(w_y, axis=0)

    w_n = []
    for xx in test_data[test_data_labels == 0]:
        w_n.append(som.winner(xx))  # getting the winner
    w_n = np.array(w_n)
    w_n = np.unique(w_n, axis=0)

    fig = plt.figure(figsize=(6, 6))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    if len(w_y) > 0:
        plt.scatter(w_y[:, 0] + .5, w_y[:, 1] + .5, color='red', marker='s', s=50, label='RCG', alpha=0.2)
    if len(w_n) > 0:
        plt.scatter(w_n[:, 0] + .5, w_n[:, 1] + .5, color='blue', marker='o', s=30, label='RSG', alpha=0.2)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.tight_layout()

    return fig


def generate_model_metrics(train_filtered, train_label, som, test_data, test_data_labels, scaler):
    def classify(som, data):
        winmap = som.labels_map(train_filtered, train_label)
        default_class = np.sum(list(winmap.values())).most_common()[0][0]
        result = []
        for d in data:
            win_position = som.winner(d)
            if win_position in winmap:
                result.append(winmap[win_position].most_common()[0][0])
            else:
                result.append(default_class)
        return result

    test_data_scaled = scaler.transform(test_data)
    som_classification = classify(som, test_data_scaled)

    model_metrics = {"f-score": metrics.f1_score(test_data_labels.values, som_classification),
                     "matthews_corrcoef": metrics.matthews_corrcoef(test_data_labels.values, som_classification)}

    return model_metrics


def get_som(train_data, train_data_labels, valid_data, valid_data_labels, test_data, test_data_labels, parameters):

    with mlflow.start_run():
        mlflow.log_params(parameters)

        scaler = MinMaxScaler()

        train_scaled = scaler.fit_transform(train_data)

        if parameters['do_ng_filtering']:
            filtered = filtro_ngbatch_list(train_scaled,
                                           train_data_labels.values,
                                           parameters['tol'],
                                           parameters['n_batches'])
        else:
            filtered = train_scaled

        train_filtered = filtered[:, :-1]
        train_label = filtered[:, -1]

        som = MiniSom(parameters['nm'],
                      parameters['nm'],
                      train_filtered.shape[1],
                      sigma=parameters['sigma'],
                      learning_rate=parameters['lr'],
                      neighborhood_function='gaussian',
                      activation_distance='euclidean',
                      random_seed=0)

        som.train(train_filtered,
                  num_iteration=parameters['num_iteration'],
                  use_epochs=parameters['use_epochs'],
                  verbose=True)

        train_figure = generate_train_map_figure(train_filtered, train_label, som)

        mlflow.log_figure(train_figure, "train_figure.png")

        test_figure = generate_test_map_figure(test_data.values, test_data_labels.values, som)

        mlflow.log_figure(test_figure, "test_figure.png")

        model_metrics = generate_model_metrics(train_filtered, train_label, som, test_data, test_data_labels, scaler)

        mlflow.log_metrics(model_metrics)


def main(data_folder_path, params_path):
    train_data = pd.read_csv(os.path.join(data_folder_path, 'TRAIN.csv'))
    valid_data = pd.read_csv(os.path.join(data_folder_path, 'VALIDATION.csv'))
    test_data = pd.read_csv(os.path.join(data_folder_path, 'TEST.csv'))

    train_data, train_data_labels = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    valid_data, valid_data_labels = valid_data.iloc[:, :-1], valid_data.iloc[:, -1]
    test_data, test_data_labels = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    with open(params_path) as f:
        parameters = json.load(f)

    get_som(train_data=train_data,
            train_data_labels=train_data_labels,
            valid_data=valid_data,
            valid_data_labels=valid_data_labels,
            test_data=test_data,
            test_data_labels=test_data_labels,
            parameters=parameters)


if __name__ == "__main__":
    data_folder_path = 'data'
    params_path = 'src/params.json'
    main(data_folder_path=data_folder_path, params_path=params_path)
