import math
from joblib import Parallel, delayed
from collections import Counter

import pandas as pd
import warnings
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

warnings.simplefilter(action='ignore', category=FutureWarning)

#method to return datasets
def get_data_sets(file_name):
    data_sets = pd.read_csv(file_name, sep=',')
    #this data set is for loan.csv
    #data_sets.columns = ['attr1_cat', 'attr2_cat', 'attr3_cat', 'attr4_cat', 'attr5_cat','attr6_num', 'attr7_num', 'attr8_num', 'attr9_num', 'attr10_cat', 'attr11_cat', 'label']

    # this data set is for wdbc.csv
    # data_sets.columns = ['attr1_num', 'attr2_num', 'attr3_num', 'attr4_num', 'attr5_num',
    #                  'attr6_num', 'attr7_num', 'attr8_num', 'attr9_num', 'attr10_num',
    #                  'attr11_num', 'attr12_num', 'attr13_num', 'attr14_num', 'attr15_num',
    #                  'attr16_num', 'attr17_num', 'attr18_num', 'attr19_num', 'attr20_num',
    #                  'attr21_num', 'attr22_num', 'attr23_num', 'attr24_num', 'attr25_num',
    #                  'attr26_num', 'attr27_num', 'attr28_num', 'attr29_num', 'attr30_num',
    #                  'label']

    # this data set is for raisin.csv
    # data_sets.columns = ['attr1_num', 'attr2_num', 'attr3_num', 'attr4_num', 'attr5_num',
    #                  'attr6_num', 'attr7_num', 'label']

    # this data set is for titanic.csv
    data_sets.columns = ['label', 'attr1_cat', 'attr2_cat', 'attr3_num', 'attr4_num', 'attr5_num',
                     'attr6_num']
    data_sets = data_sets.drop(index=0).reset_index(drop=True)
    return data_sets

# Option1: method to calculate entropy
# def calculate_entropy(data_sets):
#     entropy = 0
#     count_values = data_sets.value_counts()
#     total_counts = len(data_sets)
#     probability = count_values / total_counts
#     for prob in probability:
#         if prob > 0:
#             entropy += prob * math.log(prob, 2)
#     return -entropy

# Option2: method to calculate entropy to save time
def calculate_entropy(data_sets):
    count_values = np.bincount(data_sets.astype(int))
    probability = count_values / count_values.sum()
    probability = probability[probability > 0]  # Remove zeros
    return -np.sum(probability * np.log2(probability))

# method to calculate information gain
def calculate_information(attribute, data_sets, end_values, is_numerical=False):
    weighted_entropy = 0
    total_target_values = data_sets[end_values]
    total_entropy = calculate_entropy(total_target_values)

    if is_numerical:
        # Ensure numerical sorting and unique thresholds
        sorted_data = data_sets.sort_values(attribute).reset_index(drop=True)
        unique_values = sorted_data[attribute].unique()

        # If there's only one unique value, no split is possible
        if len(unique_values) <= 1:
            return 0, None

        # Compute possible split points (thresholds)
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        max_gain = -float('inf')
        best_threshold = None

        for threshold in thresholds:
            left_split = data_sets[data_sets[attribute] <= threshold]
            right_split = data_sets[data_sets[attribute] > threshold]

            # Ensure we have meaningful splits
            if left_split.empty or right_split.empty:
                continue

            left_entropy = calculate_entropy(left_split[end_values])
            right_entropy = calculate_entropy(right_split[end_values])

            weighted_entropy = (len(left_split) / len(data_sets)) * left_entropy + \
                               (len(right_split) / len(data_sets)) * right_entropy
            gain = total_entropy - weighted_entropy

            if gain > max_gain:
                max_gain = gain
                best_threshold = threshold

        return max_gain, best_threshold
    else:
        # For categorical attributes
        unique_values = data_sets[attribute].unique()
        weighted_entropy = 0

        for value in unique_values:
            subset = data_sets[data_sets[attribute] == value]
            if subset.empty:
                continue
            weighted_entropy += (len(subset) / len(data_sets)) * calculate_entropy(subset[end_values])

        information_gain = total_entropy - weighted_entropy
        return information_gain, None


# method to calculate the maximum information gain for an attribute
def select_information_gain(data_sets, end_values, attributes):
    best_gain_attribute = ''
    maximum_gain = 0
    best_threshold = None

    for attribute in attributes:
        # Check if attribute is numerical or categorical
        is_numerical = data_sets[attribute].dtype in ['int64', 'float64']

        attribute_information_gain, threshold = calculate_information(attribute, data_sets, end_values, is_numerical)

        if attribute_information_gain > maximum_gain:
            maximum_gain = attribute_information_gain
            best_gain_attribute = attribute
            best_threshold = threshold

    return best_gain_attribute, best_threshold


# Select the best attribute for splitting using random selection of m attributes.
def select_information_gain_randomly(data_sets, end_values, attributes):
    m = round(math.sqrt(len(attributes)))  # we are taking square root of m here( it should come 3 as there are 11 attributes for load.csv)
    selected_attributes = np.random.choice(attributes, m, replace=False)  # Randomly select m attributes
    best_gain_attribute = None
    best_threshold = None
    maximum_gain = -float('inf')

    for attribute in selected_attributes:
        is_numerical = data_sets[attribute].dtype in ['int64', 'float64']
        attribute_information_gain, threshold = calculate_information(attribute, data_sets, end_values, is_numerical)

        if attribute_information_gain > maximum_gain:
            maximum_gain = attribute_information_gain
            best_gain_attribute = attribute
            best_threshold = threshold  # Store the threshold

    return best_gain_attribute, best_threshold


def calculate_decision_tree(end_values, data_sets, attributes, max_depth=7, depth=0, min_gain=1,
                            min_samples_split=3):
    #Stop if the number of samples in the current split is less than min_samples_split
    # if len(data_sets) < min_samples_split:
    #     return data_sets[end_values].mode()[0]

    #best_attribute, threshold = select_information_gain(data_sets, end_values, attributes)
    best_attribute, threshold = select_information_gain_randomly(data_sets, end_values, attributes)

    #If the dataset has only one class left, return it as a leaf
    if len(data_sets[end_values].drop_duplicates()) == 1:
        return data_sets[end_values].iloc[0]

    #Using max depth to prevent overfitting
    if depth >= max_depth:
        value = data_sets[end_values].mode()
        if len(value) != 0:
            return value.iloc[0]
        else:
            return None

    # If no attributes left to split, return the most common class
    if len(attributes) == 0:
        return data_sets[end_values].value_counts().idxmax()

    sub_decision_tree = {best_attribute: {}}

    if threshold is not None:
        # Numerical splitting based on threshold
        left_split = data_sets[data_sets[best_attribute] <= threshold]
        right_split = data_sets[data_sets[best_attribute] > threshold]

        left_gain = calculate_information(best_attribute, left_split, end_values)[0]
        right_gain = calculate_information(best_attribute, right_split, end_values)[0]

        #If the information gain is below the minimum threshold, stop splitting
        # if left_gain < min_gain and right_gain < min_gain:
        #     return data_sets[end_values].mode()[0]

        sub_decision_tree[best_attribute]["<=" + str(threshold)] = calculate_decision_tree(end_values, left_split,
                                                                                           attributes, max_depth,
                                                                                           depth + 1)
        sub_decision_tree[best_attribute][">" + str(threshold)] = calculate_decision_tree(end_values, right_split,
                                                                                          attributes, max_depth,
                                                                                          depth + 1)
    else:
        # Categorical splitting
        for value in data_sets[best_attribute].drop_duplicates():
            subset = data_sets[data_sets[best_attribute] == value]
            sub_decision_tree[best_attribute][value] = calculate_decision_tree(end_values, subset, attributes,
                                                                               max_depth, depth + 1)

    return sub_decision_tree


def majority_voting(forest, instance):
    predictions = []
    for tree in forest:
        prediction = predict(tree, instance)
        predictions.append(prediction)

    counter = Counter(predictions) # counts the occurance of each predicted table
    majority_class = counter.most_common(1)[0][0] # returns the most frequent label.
    return majority_class


# def majority_voting(forest, row):
#     votes = [tree.predict([row])[0] for tree in forest]
#     return max(set(votes), key=votes.count)


def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), "The true and predicted labels must have the same length"
    correct_predictions = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
    return correct_predictions / len(y_true)


def stratified_k_fold_split(data, target_variable, k=5):
    grouped = data.groupby(target_variable)
    stratified_folds = [[] for _ in range(k)]

    # Split each class group into k folds
    for class_label, group in grouped:
        #shuffled_group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        shuffled_group = group.sample(frac=1, random_state=42)
        class_folds = np.array_split(shuffled_group, k)

        # Distribute the data into folds
        for i in range(k):
            stratified_folds[i].append(class_folds[i])

    # Convert list of folds into DataFrames
    final_folds = [pd.concat(fold).reset_index(drop=True) for fold in stratified_folds]

    # Print fold distribution for debugging purposes
    for fold_index, fold in enumerate(final_folds):
        print(f"Fold {fold_index} - Test data indices: {fold.index.tolist()[:5]}...")  # Check the first 5 rows

    return final_folds


def evaluate_random_forest(data, attributes, target_variable, k=5, ntree_values=None):
    if ntree_values is None:
        ntree_values = [1, 5, 10, 20, 30, 40, 50]

    accuracies = []
    recall_average = []
    precision_average = []
    f1_average = []

    X = data[attributes]  # Features
    y = data[target_variable]  # Target variable

    # Stratified K-fold initialization
    stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for ntree in ntree_values:
        fold_accuracies = []
        fold_recalls = []
        fold_precisions = []
        fold_f1_scores = []

        # Stratified K-Fold cross-validation loop
        for train_index, test_index in stratified_kfold.split(X, y):
            # Split data into training and test sets
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            # Ensure no overlap between train and test data
            overlap = test_data.index.isin(train_data.index).sum()
            if overlap > 0:
                print(f"Overlap found in fold with {overlap} overlapping indices!")
                print("Overlapping indices:", test_data.index[test_data.index.isin(train_data.index)])

            # Train Random Forest (Parallelizing the training of trees)
            forest = Parallel(n_jobs=-1)(
                delayed(calculate_decision_tree)(target_variable, train_data.sample(frac=1, replace=True), attributes)
                for _ in range(ntree)
            )

            # Make predictions on the test set
            test_predictions = [majority_voting(forest, row) for _, row in test_data.iterrows()]

            # Evaluate performance
            accuracy = accuracy_score(test_data[target_variable], test_predictions)
            recall = calculate_recall(test_data[target_variable], test_predictions)
            precision = calculate_precision(test_data[target_variable], test_predictions)
            f1 = calculate_f1(test_data[target_variable], test_predictions)

            # Append results for this fold
            fold_accuracies.append(accuracy)
            fold_recalls.append(recall)
            fold_precisions.append(precision)
            fold_f1_scores.append(f1)

        # Average the metrics across folds
        accuracies.append(np.mean(fold_accuracies))
        recall_average.append(np.mean(fold_recalls))
        precision_average.append(np.mean(fold_precisions))
        f1_average.append(np.mean(fold_f1_scores))

    plot_metrics(ntree_values, accuracies, recall_average, precision_average, f1_average)

    print("Final Accuracy for different ntree values:", accuracies)
    print("Final Recall for different ntree values:", recall_average)
    print("Final Precision for different ntree values:", precision_average)
    print("Final F1 Score for different ntree values:", f1_average)

    return accuracies, recall_average, precision_average, f1_average

def calculate_recall(Y_true, Y_pred):
    TP, FP, FN, TN = confusion_matrix(Y_true, Y_pred)
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    if(TP + FN > 0):
        recall_value = TP/float(TP+FN)
    else:
        recall_value = 0
    return recall_value

#Method to calculate the confusion matrix
def confusion_matrix(Y_true, Y_pred):
    TP = 0
    FP =0
    TN = 0
    FN = 0
    for true, pred, in zip(Y_true, Y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1
        elif true == 0 and pred == 0:
            TN += 1
    return TP, FP, FN, TN

#method to calculate prediction
def calculate_precision(Y_true, Y_pred):
    TP, FP, FN, TN = confusion_matrix(Y_true, Y_pred)
    if(TP + FP > 0):
        precision_value = TP/ float(TP+FP)
    else:
        precision_value = 0
    return precision_value

def calculate_f1(Y_true, Y_pred):
    beta = 1
    F1=0
    TP, FP, FN, TN = confusion_matrix(Y_true, Y_pred)
    precision_value = calculate_precision(Y_true, Y_pred)
    recall_value = calculate_recall(Y_true, Y_pred)
    if(precision_value + recall_value > 0):
        # value = 1+(beta**2)
        # denominator = ((beta**2 * precision_value) + recall_value)
        # numerator = (1+(beta**2))*(precision_value * recall_value)
        # f1 = numerator/denominator

        F1 = 2 * precision_value * recall_value / (precision_value + recall_value) # for same weights for precision and recall
    return F1

def extract_threshold(key):
    """Extract numeric value from threshold key (handles '<=value' and '>value')."""
    try:
        return float(key.lstrip('<>='))  # Remove '<=', '>', or '=' and convert
    except ValueError:
        return float('inf')  # Fallback for unexpected formats

def predict(tree, instance):
    if isinstance(tree, dict):
        attribute = list(tree.keys())[0]
        attribute_value = instance[attribute]

        if isinstance(attribute_value, (int, float)):
            threshold_keys = list(tree[attribute].keys())

            def extract_threshold(key):
                return float(key.lstrip('<>=')) if isinstance(key, str) else float(key)

            threshold_values = [extract_threshold(k) for k in threshold_keys]

            if attribute_value <= min(threshold_values):
                key = f"<={min(threshold_values)}"
            else:
                key = f">{max(threshold_values)}"

            subtree = tree[attribute].get(key, list(tree[attribute].values())[0])
        else:
            subtree = tree[attribute].get(attribute_value, list(tree[attribute].values())[0])

        return predict(subtree, instance)

    return tree if isinstance(tree, (str, int, float, np.int64)) else list(tree.keys())[0]

def plot_metrics(ntree_values, accuracies, precisions, recalls, f1_scores):
    # Helper function to set appropriate y-axis limits
    def get_limits(values):
        return min(values) - 0.02, max(values) + 0.02  # Small buffer for visibility

    # Get limits for each metric
    acc_min, acc_max = get_limits(accuracies)
    prec_min, prec_max = get_limits(precisions)
    rec_min, rec_max = get_limits(recalls)
    f1_min, f1_max = get_limits(f1_scores)

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(ntree_values, accuracies, marker='o', label='Accuracy', color='blue', markersize=8)
    plt.xlabel('Number of Trees (ntree)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Trees')
    plt.ylim(acc_min, acc_max)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Precision
    plt.figure(figsize=(10, 6))
    plt.plot(ntree_values, precisions, marker='s', label='Precision', color='green', markersize=8)
    plt.xlabel('Number of Trees (ntree)')
    plt.ylabel('Precision')
    plt.title('Precision vs. Number of Trees')
    plt.ylim(prec_min, prec_max)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Recall
    plt.figure(figsize=(10, 6))
    plt.plot(ntree_values, recalls, marker='^', label='Recall', color='red', markersize=8)
    plt.xlabel('Number of Trees (ntree)')
    plt.ylabel('Recall')
    plt.title('Recall vs. Number of Trees')
    plt.ylim(rec_min, rec_max)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(ntree_values, f1_scores, marker='d', label='F1 Score', color='purple', markersize=8)
    plt.xlabel('Number of Trees (ntree)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Number of Trees')
    plt.ylim(f1_min, f1_max)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    filename = 'loan.csv'
    filename1 = 'wdbc.csv'
    filename2 = 'raisin.csv'
    filename3 = 'titanic.csv'
    filepath = '/../../Users/amitkumar/Documents/ML/HW3_CMPSCI_589_Spring2025_Supporting_Files/'
    #for loan.csv
    data_sets_attributes = ['attr1_cat', 'attr2_cat', 'attr3_cat', 'attr4_cat', 'attr5_cat',
                            'attr6_num', 'attr7_num', 'attr8_num', 'attr9_num', 'attr10_cat', 'attr11_cat']

    #for wdbc.csv
    data_sets_attributes_for_wdbc = ['attr1_num', 'attr2_num', 'attr3_num', 'attr4_num', 'attr5_num',
                     'attr6_num', 'attr7_num', 'attr8_num', 'attr9_num', 'attr10_num',
                     'attr11_num', 'attr12_num', 'attr13_num', 'attr14_num', 'attr15_num',
                     'attr16_num', 'attr17_num', 'attr18_num', 'attr19_num', 'attr20_num',
                     'attr21_num', 'attr22_num', 'attr23_num', 'attr24_num', 'attr25_num',
                     'attr26_num', 'attr27_num', 'attr28_num', 'attr29_num', 'attr30_num']

    # for raisin.csv
    data_sets_attributes_for_raisin = ['attr1_num', 'attr2_num', 'attr3_num', 'attr4_num', 'attr5_num',
                     'attr6_num', 'attr7_num']

    #for titanic.csv
    data_sets_attributes_for_titanic = ['attr1_cat', 'attr2_cat', 'attr3_num', 'attr4_num', 'attr5_num',
                         'attr6_num']
    data = get_data_sets(filepath+filename3)
    test_data = data.head(100)
    # test with less data
    #print(f"Evaluating attribute: {data_sets_attributes}")
    #evaluate_random_forest(data, data_sets_attributes, 'label', k=5)
    #evaluate_random_forest(data, data_sets_attributes_for_wdbc, 'label', k=5)
    evaluate_random_forest(data, data_sets_attributes_for_titanic, 'label', k=5)