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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random)

    X_test_df = pd.DataFrame(X_test, columns=['buying_price', 'maintenance_price', 'number_doors', 'capacity',
                                              'luggage_boot_size', 'safety_level'])
    X_train_df = pd.DataFrame(X_train, columns=['buying_price', 'maintenance_price', 'number_doors', 'capacity',
                                                'luggage_boot_size', 'safety_level'])

    y_train_df = pd.Series(y_train)
    y_test_df = pd.Series(y_test)

    return X_train_df, y_train_df, X_test_df, y_test_df

# method to calculate the gini impurity

def calculate_gini(data_sets):

    gini_coeffcient = 0
    count_values = data_sets.value_counts()
    total_counts = len(data_sets)

    probability = count_values / total_counts
    for prob in probability:
        total_probability = prob **2
        gini_coeffcient = gini_coeffcient - total_probability
    return gini_coeffcient

# method to calculate the gini coefficient

def calculate_gini_coefficient(attribute, data_sets, end_values):
    weighted_gini_coefficient = 0
    total_target_values = data_sets[end_values]
    unique_total_values = set(data_sets[attribute])

    total_gini_coefficient = calculate_gini(total_target_values)

    for value in unique_total_values:
        subset = data_sets[data_sets.get(attribute) == value]
        subset_target_values = subset.get(end_values)
        subset_gini_coefficient = calculate_gini(subset_target_values)
        weighted_gini_coefficient += (len(subset) / len(data_sets)) * subset_gini_coefficient
    gini_value = total_gini_coefficient - weighted_gini_coefficient
    return gini_value

# calculate to select with the lowest gini
def select_minium_gini_value(data_sets, end_values, attributes):
    best_gini_coefficient_value= ''
    minimum = float('inf')
    for single_attribute in attributes:
        attribute_gini_value = calculate_gini_coefficient(single_attribute, data_sets, end_values)

        if attribute_gini_value < minimum:
            minimum = attribute_gini_value
            best_gini_coefficient_value = single_attribute

    return best_gini_coefficient_value

# method to calculate the decision trees

def calculate_decision_tree(end_values, data_sets, attributes, max_depth, depth):
    new_attrbutes = []

    best_attribute = select_minium_gini_value(data_sets, end_values, attributes)
    for a in attributes:
        if a!= best_attribute:
            new_attrbutes.append(a)
    sub_decision_tree = {best_attribute: {}}

    # If the dataset has only one class left, return it as a leaf( stopping criteria1)
    if len(data_sets[end_values].drop_duplicates()) == 1:
        return data_sets.get(end_values).iloc[0]

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

def run_experiment_for_training_sets(data_sets, number_experiments, max_depth, depth):
    number_experiments = number_experiments
    training_accuracies = []
    for i in range(number_experiments):

        X_train_df, y_train_df, X_test_df, y_test_df = split_datasets(data_sets, i)
        decision_tree_model = calculate_decision_tree('class',data_sets,X_train_df.columns.tolist(), max_depth, depth)
        training_accuracy = calculate_accuracy(decision_tree_model, X_train_df, y_train_df)

        training_accuracies.append(training_accuracy)

    return training_accuracies

def run_experiment_for_testing_sets(data_sets, number_experiments, max_depth, depth):
    number_experiments = number_experiments
    test_accuracies = []

    for i in range(number_experiments):

        X_train_df, y_train_df, X_test_df, y_test_df = split_datasets(data_sets, i)

        decision_tree_model = calculate_decision_tree('class', data_sets,X_train_df.columns.tolist(), max_depth = max_depth, depth=depth)
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
    random_number = 100
    number_experiments = 100

    X_train_df, y_train_df, X_test_df, y_test_df = split_datasets(data, random_number)

    output_class = ['class']
    x_attributes = ['buying_price', 'maintenance_price', 'number_doors', 'capacity', 'luggage_boot_size',
                    'safety_level']

    decision_tree = calculate_decision_tree(output_class, data, x_attributes, None, 0)
    training_datasets_accuracy = calculate_accuracy(decision_tree, X_train_df, y_train_df)
    test_datasets_accuracy = calculate_accuracy(decision_tree, X_test_df, y_test_df)

    print(
        f'Training Data Sets Accuracy: {training_datasets_accuracy} and Test Data Sets Accuracy: {test_datasets_accuracy}')

    train_accuracy = run_experiment_for_training_sets(data, 100, None, 0)
    test_accuracy = run_experiment_for_testing_sets(data, 100, None, 0)
    print(train_accuracy, "training accuracy")
    print(test_accuracy, "test accuracy")

    print("mean of training datasets accuracy:", np.mean(train_accuracy))
    print("mean of test datasets accuracy:", np.mean(test_accuracy))
    print("Standard Deviation of train datasets accuracy:", np.std(train_accuracy))
    print("standard deviation of test datasets accuracy:", np.std(test_accuracy))
    plot_histogram_for_decision_tree(train_accuracy, "Training Accuracy among 100 for training datasets")
    plot_histogram_for_decision_tree(test_accuracy, "Training Accuracy among 100 for test datasets")