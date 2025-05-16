from _csv import reader
from math import sqrt
import random

import matplotlib.pyplot as plt
import numpy as np

# method to split the data_sets
def split_data_sets(data_sets, split_ratio):

    random.shuffle(data_sets)

    train_size = int(len(data_sets) * split_ratio)
    test_data_sets = data_sets[train_size:]
    train_data_sets = data_sets[:train_size]

    return test_data_sets, train_data_sets


# method to read csv file
def read_csv_file(filename):
    data_set = list()

    with open(filename, newline='') as csvfile:
        csvreader = reader(csvfile)
        for row in csvreader:
            if row:
                for i in range(len(row)):
                    row[i] = float(row[i])
                row[-1] = int(row[-1])
                data_set.append(row)

    return data_set

# method to do normalization using maximum and minimum
def normalization(data_sets):

    data_sets = np.array(data_sets)

    normalized_data_sets = (data_sets - np.min(data_sets, axis=0)) / (np.max(data_sets, axis=0) - np.min(data_sets, axis=0))
    return normalized_data_sets

# method to calculate the distance
def calculate_euclidean_distance(row1, row2):

    distance_calcuated = 0.0

    for i in range(len(row1)-1):

        distance_calcuated += ((row1[i] - row2[i]) ** 2)
    return sqrt(distance_calcuated)

def get_first_element(x):
    return x[0]

# method to find neighbors
def get_neighbors(train_data, test_row_data, neighbors):

    distances_calculated = list()
    neighbors_data = list()

    for train_row_data in train_data:
        distance = (calculate_euclidean_distance(train_row_data, test_row_data))
        distances_calculated.append((distance, train_row_data))

    distances_calculated.sort(key=get_first_element)

    for i in range(neighbors):
        value = distances_calculated[i][1]
        neighbors_data.append(value)

    return neighbors_data

# method to predict
def prediction(train, test_row, neighbors):
    count = {}

    neighbors = get_neighbors(train, test_row, neighbors)
    result_value = [row[-1] for row in neighbors]

    for value in result_value:
        if value in count:
            count[value] += 1
        else:
            count[value] = 1

    prediction = max(count, key=count.get)

    return prediction

#method to execute knn algorithm using k
def evaluate_algorithm(train_data, test_data, neighbors):
    correct = 0

    for row in test_data:

        prediction_result = prediction(train_data, row, neighbors)
        if prediction_result == row[-1]:
            correct += 1
        accuracy = (correct / len(test_data))*100
    return accuracy

#method to plot the graph
def plot_graph_KNN(accuracies, title, k_range):

    plt.errorbar(k_range, accuracies[:, 0],yerr=accuracies[:, 1], fmt='o', label=title)

    plt.ylabel('Accuracy')
    plt.xlabel('k values')
    plt.title(f'{title} accuracy vs k')

    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    filename = '/../../Users/amitkumar/Documents/ML/HW1_CMPSCI_589_Spring2025_Supporting_Files/datasets/wdbc.csv'
    data_set = read_csv_file(filename)
    normalized_data_sets = normalization(data_set)
    test_set, train_set = split_data_sets(normalized_data_sets, 0.8)
    unormalized_test_sets, unormalized_train_sets = split_data_sets(normalized_data_sets, 0.8)
    range_k = list(range(1, 51, 2))

    accuracy_train_datasets = []
    accuracy_test_datasets = []
    unormalized_accuracy_train_datasets = []
    unormalized_accuracy_test_datasets = []

    train_datasets_mean_accuracy = []
    test_datasets_mean_accuracy = []
    unormalized_train_accuracy = []
    unormalized_test_accuracy = []

    train_datasets_std_accuracy = []
    test_datasets_std_accuracy = []
    unormalized_accuracy_test_sets = []
    unormalized_accuracy_train_sets = []

    for k in range(1, 51):
        if(k%2!=0):
            train_accuracy = evaluate_algorithm(train_set, train_set, k)
            accuracy_train_datasets.append(train_accuracy)

            test_accuracy = evaluate_algorithm(test_set, test_set, k)
            accuracy_test_datasets.append(test_accuracy)

            unormalized_train_accuracy = evaluate_algorithm(unormalized_train_sets, unormalized_train_sets, k)
            unormalized_accuracy_train_datasets.append(unormalized_train_accuracy)

            unormalized_test_accuracy = evaluate_algorithm(unormalized_test_sets, unormalized_test_sets, k)
            unormalized_accuracy_test_datasets.append(unormalized_test_accuracy)

            if (len(accuracy_train_datasets) > 0) and len(accuracy_test_datasets) > 0:
                train_datasets_mean_accuracy = accuracy_train_datasets
                test_datasets_mean_accuracy = accuracy_test_datasets
                unormalized_train_accuracy = unormalized_accuracy_train_datasets
                unormalized_test_accuracy = unormalized_accuracy_test_datasets


            train_datasets_std_accuracy = [np.std(unormalized_accuracy_train_datasets)] * len(range_k)
            test_datasets_std_accuracy = [np.std(unormalized_accuracy_test_datasets)] * len(range_k)
            unormalized_accuracy_test_sets = [np.std(unormalized_accuracy_test_datasets)] * len(range_k)
            unormalized_accuracy_train_sets = [np.std(unormalized_accuracy_train_datasets)] * len(range_k)

    plot_graph_KNN(np.column_stack((test_datasets_mean_accuracy, test_datasets_std_accuracy)), 'Test sets', range_k)
    plot_graph_KNN(np.column_stack((train_datasets_mean_accuracy, train_datasets_std_accuracy)), 'Training sets', range_k)
    plot_graph_KNN(np.column_stack((unormalized_test_accuracy, unormalized_accuracy_test_sets)), 'non-normalized Training sets',
    range_k)
    plot_graph_KNN(np.column_stack((unormalized_train_accuracy, unormalized_accuracy_train_sets)), 'non-normalized Test sets',
    range_k)

