import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# method to import csv file to get the data sets

def get_data_sets(file_name):

    data_sets = pd.read_csv(file_name, sep=',', header=None)
    data_sets_attributes = ['buying_price', 'maintenance_price', 'number_doors', 'capacity', 'luggage_boot_size',
                            'safety_level', 'class']
    data_sets.columns = data_sets_attributes

    data_sets = data_sets.drop(index=0).reset_index(drop=True)
    return data_sets

# method to split the datasets into training and testing datasets and select random 100 datasets

def split_datasets(data_sets, random):

    X = data_sets.values[:, 0:6]
    Y = data_sets.values[:, 6]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=random)

    X_test_df = pd.DataFrame(X_test, columns=['buying_price', 'maintenance_price', 'number_doors', 'capacity',
                                              'luggage_boot_size', 'safety_level'])
    X_train_df = pd.DataFrame(X_train, columns=['buying_price', 'maintenance_price', 'number_doors', 'capacity',
                                                'luggage_boot_size', 'safety_level'])

    y_train_df = pd.Series(y_train)
    y_test_df = pd.Series(y_test)

    return X_train_df, y_train_df, X_test_df, y_test_df

# method to calculate the entropy

def calculate_entropy(data_sets):

    entropy = 0
    count_values = data_sets.value_counts()
    total_counts = len(data_sets)

    probability = count_values / total_counts
    for prob in probability:
        entropy += prob * math.log(prob, 2)
    return -entropy

# method to calculate the information gain for attribute

def calculate_information(attribute, data_sets, end_values):

    weighted_entropy = 0
    total_target_values = data_sets[end_values]
    unique_total_values = set(data_sets[attribute])

    total_entropy = calculate_entropy(total_target_values)

    for value in unique_total_values:
        subset = data_sets[data_sets.get(attribute) == value]
        subset_target_values = subset.get(end_values)
        subset_entropy = calculate_entropy(subset_target_values)
        weighted_entropy+= (len(subset)/len(data_sets))* subset_entropy
    information_gain = total_entropy - weighted_entropy
    return information_gain

# method to calculate the maximum information gain for attribute

def select_information_gain(data_sets, end_values, attributes):

    best_gain_attribute = ''
    maximum_gain = 0
    for single_attribute in attributes:
        attribute_information_gain = calculate_information(single_attribute, data_sets, end_values)

        if attribute_information_gain > maximum_gain:

            maximum_gain = attribute_information_gain
            best_gain_attribute = single_attribute

    return best_gain_attribute

# method to calculate the decision trees

def calculate_decision_tree(end_values, data_sets, attributes, max_depth=0, depth=0):
    new_attrbutes = []

    best_attribute = select_information_gain(data_sets, end_values, attributes)
    for a in attributes:
        if a!= best_attribute:
            new_attrbutes.append(a)
    sub_decision_tree = {best_attribute: {}}

    # If the dataset has only one class left, return it as a leaf( stopping criteria1)
    if len(data_sets[end_values].drop_duplicates()) == 1:
        return data_sets.get(end_values).iloc[0]

    # if depth >= max_depth:
    #     value = data_sets.get(end_values).mode()
    #
    #     if len(value) != 0:
    #         return value.iloc[0]
    #     else:
    #         return None

    # If no attributes left to split, return the most common class( stopping criteria 2)
    if len(attributes) == 0:
        return data_sets.get(end_values).value_counts().idxmax()

    for value in data_sets.get(best_attribute).drop_duplicates():
        attribute = data_sets[data_sets.get(best_attribute) == value]

        if len(attribute) == 0:
            sub_decision_tree.get(best_attribute)[value] = data_sets.get(end_values).mode()[0]
        else:
            sub_decision_tree.get(best_attribute)[value] = calculate_decision_tree(end_values, attribute, new_attrbutes, max_depth, depth + 1)

    return sub_decision_tree

# method to do prediction

def calculate_prediction(row_values, decision_tree):

    while type(decision_tree) == dict:

        attribute = list(decision_tree.keys())[0]
        if attribute in row_values:
            attrbute_values = row_values.get(attribute)
            if attrbute_values in decision_tree.get(attribute):
                decision_tree = decision_tree[attribute].get(attrbute_values)

    return decision_tree

# method to compare the test data and calculate the accuracy

def calculate_accuracy(decision_tree, X_data_sets, y_data_sets):

    total_number_of_instances = len(X_data_sets)
    correct_predictions = 0

    for i in range(total_number_of_instances):

        row_values = {}
        row = X_data_sets.iloc[i]
        for attribute in X_data_sets.columns:
            row_values[attribute] = row.get(attribute)

        actual_class = y_data_sets.iloc[i]
        actual_class = actual_class if isinstance(actual_class, str) else actual_class.item()

        predicted_class = calculate_prediction(row_values, decision_tree)

        predicted_class = predicted_class if isinstance(predicted_class, str) else predicted_class.item()

        if predicted_class == actual_class:
            correct_predictions += 1

    accuracy = (correct_predictions / total_number_of_instances) * 100
    return accuracy

def run_experiment_for_training_sets(data_sets, number_experiments):
    number_experiments = 100
    training_accuracies = []
    for i in range(number_experiments):

        X_train_df, y_train_df, X_test_df, y_test_df = split_datasets(data_sets, i)
        decision_tree_model = calculate_decision_tree('class',data_sets,X_train_df.columns.tolist())
        training_accuracy = calculate_accuracy(decision_tree_model, X_train_df, y_train_df)

        training_accuracies.append(training_accuracy)

    return training_accuracies

def run_experiment_for_testing_sets(data_sets, number_experiments):
    number_experiments = 100
    test_accuracies = []

    for i in range(number_experiments):

        X_train_df, y_train_df, X_test_df, y_test_df = split_datasets(data_sets, i)

        decision_tree_model = calculate_decision_tree('class', data_sets,X_train_df.columns.tolist())
        test_accuracy = calculate_accuracy(decision_tree_model, X_test_df, y_test_df)

        test_accuracies.append(test_accuracy)

    return test_accuracies

def plot_histogram_for_decision_tree(accuracy, title):
    plt.figure(figsize=(12, 8))

    plt.hist(accuracy, bins=20, edgecolor='black', alpha=0.7, color = 'skyblue')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency on accuracy')
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    filename = '/../../Users/amitkumar/Documents/ML/HW1_CMPSCI_589_Spring2025_Supporting_Files/datasets/car.csv'
    data = get_data_sets(filename)
    random_number = 42
    number_experiments = 100

    X_train_df, y_train_df, X_test_df, y_test_df  = split_datasets(data, random_number)

    output_class = ['class']
    x_attributes = ['buying_price', 'maintenance_price', 'number_doors', 'capacity', 'luggage_boot_size', 'safety_level']

    decision_tree = calculate_decision_tree(output_class, data, x_attributes)
    training_datasets_accuracy = calculate_accuracy(decision_tree, X_train_df, y_train_df)
    test_datasets_accuracy = calculate_accuracy(decision_tree, X_test_df, y_test_df)

    print(f'Training Data Sets Accuracy: {training_datasets_accuracy} and Test Data Sets Accuracy: {test_datasets_accuracy}')

    train_accuracy = run_experiment_for_training_sets(data, number_experiments)
    test_accuracy = run_experiment_for_testing_sets(data, number_experiments)

    print(train_accuracy, "training accuracy")
    print(test_accuracy, "test accuracy")
   # print("mean of training datasets accuracy:", np.mean(train_accuracy))
   # print("mean of test datasets accuracy:", np.mean(test_accuracy))
   # print("Standard Deviation of train datasets accuracy:", np.std(train_accuracy))
   # print("standard deviation of test datasets accuracy:", np.std(test_accuracy))

    plot_histogram_for_decision_tree(train_accuracy, "Training Accuracy among 100 for training datasets")
    plot_histogram_for_decision_tree(test_accuracy, "Training Accuracy among 100 for test datasets")