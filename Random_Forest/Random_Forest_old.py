import math
from collections import Counter
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import warnings

from matplotlib import pyplot as plt

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


# method to calculate the information gain for attribute for numerical and categorical
def calculate_information(attribute, data_sets, end_values, is_numerical=False):

    weighted_entropy = 0
    total_target_values = data_sets[end_values]
    total_entropy = calculate_entropy(total_target_values)
    if is_numerical:
        sorted_data = data_sets.sort_values(attribute).reset_index(drop=True)
        unique_values = sorted_data[attribute].unique()
        if len(unique_values) <= 1:
            return 0, None
        thresholds = (unique_values[:-1] + unique_values[1:])/2
        best_threshold = None
        max_gain = -float('inf')
        for threshold in thresholds:
            left_split = data_sets[data_sets[attribute] <= threshold]
            right_split = data_sets[data_sets[attribute] > threshold]
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
        weighted_entropy = 0
        unique_values = data_sets[attribute].unique()
        for value in unique_values:
            subset = data_sets[data_sets[attribute] == value]
            if subset.empty:
                continue
            #subset_target_values = subset.get(end_values)
            subset_entropy = calculate_entropy(subset[end_values])
            weighted_entropy += (len(subset)/len(data_sets))* subset_entropy
        information_gain = total_entropy - weighted_entropy
        return information_gain, None

# method to calculate the maximum information gain for attribute
def select_information_gain(data_sets, end_values, attributes):
    best_threshold = None
    best_gain_attribute = ''
    maximum_gain = 0

    for single_attribute in attributes:
        is_numerical = data_sets[single_attribute].dtype in ['int64', 'float64']
        attribute_information_gain, threshold = calculate_information(single_attribute, data_sets, end_values, is_numerical)

        if attribute_information_gain > maximum_gain:

            maximum_gain = attribute_information_gain
            best_gain_attribute = single_attribute
            best_threshold = threshold

    return best_gain_attribute, best_threshold

# select maximum information gain for m random attributes
def select_information_gain_for_m_randomly(data_sets, end_values, attributes):
    best_gain_attribute = None
    best_threshold = None
    maximum_gain = -float('inf')
    m = round(math.sqrt(len(attributes))) # we are taking square root of m here
    selected_attributes = np.random.choice(attributes, m, replace=False)
    for single_attribute in selected_attributes:
        is_numerical = data_sets[single_attribute].dtype in ['int64', 'float64']
        information_gain, threshold = calculate_information(single_attribute, data_sets, end_values, is_numerical)
        if information_gain > maximum_gain:
            maximum_gain = information_gain
            best_gain_attribute = single_attribute
            best_threshold = threshold
    return best_gain_attribute, best_threshold

# method to calculate the decision trees
def calculate_decision_tree(end_values, data_sets, attributes, max_depth=4, depth=0, min_gain=0.01,
                            min_samples_split=2):
    # Stop if the number of samples in the current split is less than min_samples_split
    # if len(data_sets) < min_samples_split:
    #     return data_sets[end_values].mode()[0]

    #best_attribute, threshold = select_information_gain(data_sets, end_values, attributes)
    best_attribute, threshold = select_information_gain_for_m_randomly(data_sets, end_values, attributes)

    # If the dataset has only one class left, return it as a leaf
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

# method to give majority vote
def majority_voting(random_forest_tree, instance):
    predictions = []
    for tree in random_forest_tree:
        prediction = predict(tree, instance)
        predictions.append(prediction)
    counter = Counter(predictions) # count the occurance of each predicted value
    majority_value = counter.most_common(1)[0][0] # return the most frequent label
    return majority_value

#method to give accuracy score
def accuracy_score(Y_true, Y_predicted):
    assert len(Y_true) == len(Y_predicted)
    correct_predictions = sum([1 for true, pred in zip(Y_true, Y_predicted) if true == pred])
    return correct_predictions / len(Y_true)

# method to provide stratified folds in k
def stratified_k_fold(data_sets, end_value, k=5):
    grouped = data_sets.groupby(end_value)
    stratified_folds = [[] for i in range(k)]
    for class_label, group in grouped:
        shuffled_group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        class_folds = np.array_split(shuffled_group, k)
        for i in range (k):
            stratified_folds[i].append(class_folds[i])
    return [pd.concat(fold).reset_index(drop=True) for fold in stratified_folds]

#method to calculate recall
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

#method to calculate the f1_score
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
    try:
        return float(key.lstrip('<>='))
    except ValueError:
        return float('inf')

# method to predict
def predict(tree, instance):
    if isinstance(tree, dict):
        attribute = list(tree.keys())[0]
        attribute_value = instance[attribute]

        if isinstance(attribute_value, (int, float)):
            thresholds = list(tree[attribute].keys())

            def extract_threshold(key):
                return float(key.lstrip('<>=')) if isinstance(key, str) else float(key)

            threshold_value = [extract_threshold(k) for k in thresholds]
            if attribute_value <= min(threshold_value):
                key = f"<={min(threshold_value)}"
            else:
                key = f">={max(threshold_value)}"
            subtree = tree[attribute].get(key, list(tree[attribute].values())[0])

        else:
            subtree = tree[attribute].get(attribute_value, list(tree[attribute].values())[0])
        return predict(subtree, instance)
    return tree if isinstance(tree, (str, int, float, np.int64)) else list(tree.keys())[0]

def plot_metrics(n_tree, accuracies, precision, recall, f_score):

    def get_limits(value):
        return min(value) - 0.02, max(value) + 0.02

    accuracy_min, accuracy_max = get_limits(accuracies)
    precision_min, precision_max = get_limits(precision)
    recall_min, recall_max = get_limits(recall)
    f_score_min, f_score_max = get_limits(f_score)

    plt.figure(figsize=(10, 6))
    plt.plot(n_tree, accuracies, marker = 'o', label="Accuracy", color = 'blue', markersize=8)
    plt.xlabel('Number of trees(n_trees')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of trees(n_trees)')
    plt.ylim(accuracy_min, accuracy_max)
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(n_tree, precision, marker='o', label="Precision", color='red', markersize=8)
    plt.xlabel('Number of trees(n_trees')
    plt.ylabel('Precision')
    plt.title('Precision vs. Number of trees(n_trees)')
    plt.ylim(precision_min, precision_max)
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(n_tree, recall, marker='o', label="Recall", color='green', markersize=8)
    plt.xlabel('Number of trees(n_trees')
    plt.ylabel('Recall')
    plt.title('Recall vs. Number of trees(n_trees)')
    plt.ylim(recall_min, recall_max)
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(n_tree, f_score, marker='o', label="F1 score", color='orange', markersize=8)
    plt.xlabel('Number of trees(n_trees')
    plt.ylabel('F1 score')
    plt.title('F1 score vs. Number of trees(n_trees)')
    plt.ylim(f_score_min, f_score_max)
    plt.grid(True)
    plt.legend()
    plt.show()

def evaluate_random_forest(data_sets, attributes, end_values, n_tree=None, k=5):
    if n_tree is None:
        n_tree = [1, 5, 10, 20, 30, 40, 50]
    accuracies = []
    precisions_average = []
    recalls_average = []
    f1s_average = []

    for tree in n_tree:
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []

        stratified_folds = stratified_k_fold(data_sets, end_values, k)
        for folds in range (k):
            test_data = stratified_folds[folds]
            train_data = pd.concat([fold for i, fold in enumerate(stratified_folds) if i!=folds])

            #option 1:
            # forest = []
            # for _ in range(tree):
            #     bootstrap_sample = train_data.sample(frac=1, replace=True)
            #     tree = calculate_decision_tree(end_values, bootstrap_sample, attributes)
            #     forest.append(tree)

            # option 2 parallel execution:
            forest = Parallel(n_jobs=-1)(
                delayed(calculate_decision_tree)(end_values,
                                                 train_data.sample(frac=1, replace=True),
                                                 attributes)
                for _ in range(tree)
            )
            test_predictions = [majority_voting(forest, row) for _, row in test_data.iterrows()]
            accuracy = accuracy_score(test_data[end_values], test_predictions)
            recall = calculate_recall(test_data[end_values], test_predictions)
            precision = calculate_precision(test_data[end_values], test_predictions)
            f1 = calculate_f1(test_data[end_values], test_predictions)
            fold_accuracies.append(accuracy)
            fold_recalls.append(recall)
            fold_precisions.append(precision)
            fold_f1s.append(f1)

            # Calculate average metrics across folds
        print(f"the N-tree value is {tree}")
        accuracies.append(np.mean(fold_accuracies))
        recalls_average.append(np.mean(fold_recalls))
        precisions_average.append(np.mean(fold_precisions))
        f1s_average.append(np.mean(fold_f1s))
    plot_metrics(n_tree, accuracies, recalls_average, precisions_average, f1s_average)

    # Print final metrics
    # Calculate average metrics across folds
    print("Final Accuracy for different ntree values:", accuracies)
    print("Final Recall for different ntree values:", recalls_average)
    print("Final Precision for different ntree values:", precisions_average)
    print("Final F1 Score for different ntree values:", f1s_average)

    return accuracies, recalls_average, precisions_average, f1s_average


if __name__ == "__main__":
    filename = 'loan.csv'
    filename1 = 'wdbc.csv'
    filename2 = 'raisin.csv'
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

    data = get_data_sets(filepath+filename2)
    test_data = data.head(100) # test with less data
    #print(f"Evaluating attribute: {data_sets_attributes}")
    #evaluate_random_forest(data, data_sets_attributes, 'label', k=5)
    #evaluate_random_forest(data, data_sets_attributes_for_wdbc, 'label', k=5)
    evaluate_random_forest(data, data_sets_attributes_for_raisin, 'label', k=5)