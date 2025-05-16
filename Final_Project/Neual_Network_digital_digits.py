import time

import numpy as np
import pandas as pd
import warnings

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

warnings.simplefilter(action='ignore', category=FutureWarning)

class NeuralNetwork:
    def __init__(self, layer_sizes, seed=42, lambd=0.0):
        np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.lambd = lambd

        self.weights = []
        for i in range(self.num_layers - 1):

            weight_matrix = np.random.randn(layer_sizes[i + 1], layer_sizes[i] + 1) * np.sqrt(2 / layer_sizes[i])
            self.weights.append(weight_matrix)

    def sigmoid_function(self, z):

        z = np.array(z, dtype=np.float64)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative_function(self, a):

        return a * (1 - a)

    def forward_pass(self, input_data):
        pre_acts = []
        acts = []
        current = input_data

        for i, w in enumerate(self.weights):
            bias_row = np.ones((1, current.shape[1]))
            current_with_bias = np.vstack([bias_row, current])
            z = np.dot(w, current_with_bias)
            if i == len(self.weights) - 1:
                current = self.softmax(z)  # last layer
            else:
                current = self.sigmoid_function(z)
            pre_acts.append(z)
            acts.append(current)

        return pre_acts, acts

    def compute_cost(self, output, target):
        num_samples = target.shape[1]
        cost = -np.sum(target * np.log(output + 1e-8)) / num_samples

        if self.lambd > 0:
            reg_sum = sum(np.sum(w[:, 1:] ** 2) for w in self.weights)
            cost += (self.lambd / (2 * num_samples)) * reg_sum

        return cost

    def backward_pass(self, acts, input_data, target):
        num_samples = target.shape[1]
        grads = []
        for w in self.weights:
            grads.append(np.zeros_like(w))

        errors = [None] * (self.num_layers - 1)
        all_acts = [input_data] + acts[:-1]

        errors[-1] = acts[-1] - target

        for i in reversed(range(self.num_layers - 2)):
            next_w = self.weights[i + 1][:, 1:]
            back = np.dot(next_w.T, errors[i + 1])
            deriv = self.sigmoid_derivative_function(acts[i])
            errors[i] = back * deriv

        for i in range(self.num_layers - 1):
            a_prev = all_acts[i]
            bias_row = np.ones((1, a_prev.shape[1]))
            a_prev_with_bias = np.vstack([bias_row, a_prev])

            grad = (1 / num_samples) * np.dot(errors[i], a_prev_with_bias.T)

            reg = np.copy(self.weights[i])
            reg[:, 0] = 0
            reg_term = (self.lambd / num_samples) * reg
            grad += reg_term

            grads[i] = grad

        return grads

    def update_parameters(self, grads, learning_rate):
        #print("code is coming here in update parameter method")
        for l in range(self.num_layers - 1):

            self.weights[l] -= learning_rate * grads[l]

    def cross_validate(self, X, Y, folds=5, learning_rate=0.1, epsilon=1e-6, epochs=1000):
        f1_scores, accuracies = [], []
        Y_labels = np.argmax(Y, axis=0) if Y.shape[0] > 1 else Y.flatten()
        classes = np.unique(Y_labels)
        fold_indices = [[] for _ in range(folds)]

        for c in classes:
            idx = np.where(Y_labels == c)[0]
            np.random.shuffle(idx)
            for i, part in enumerate(np.array_split(idx, folds)):
                fold_indices[i].extend(part)

        for fold in range(folds):
            train_idx = [idx for i in range(folds) if i != fold for idx in fold_indices[i]]
            test_idx = fold_indices[fold]
            X_train, X_test = X[:, train_idx], X[:, test_idx]
            Y_train, Y_test = Y[:, train_idx], Y[:, test_idx]

            self.__init__(self.layer_sizes, lambd=self.lambd)
            self.train(X_train, Y_train, lr=learning_rate, max_epochs=epochs, tol=epsilon, verbose=False)

            predictions = self.predict(X_test)  # shape (n_samples,)
            labels = np.argmax(Y_test, axis=0)
            accuracies.append(np.mean(predictions == labels))
            f1_scores.append(self.calculate_multiclass_performance_metrics(predictions, labels))
            #f1_scores.append(self.calculate_f1(predictions, labels))

        return np.mean(accuracies), np.mean(f1_scores)

    def train(self, input_data, target, lr=0.1, max_epochs=2000, tol=1e-6, verbose=True):
        prev_cost = float('inf')

        for epoch in range(max_epochs):
            _, acts = self.forward_pass(input_data)
            grads = self.backward_pass(acts, input_data, target)
            cost_now = self.compute_cost(acts[-1], target)
            self.update_parameters(grads, lr)

            if verbose and epoch % 100 == 0:
                print("Epoch", epoch, "Cost:", cost_now)

            if abs(prev_cost - cost_now) < tol:
                if verbose:
                    print("Early stopping at epoch", epoch, "with cost", cost_now)
                break

            prev_cost = cost_now

    def predict(self, input_data):
        _, acts = self.forward_pass(input_data)
        final_output = acts[-1]  # shape (10, n_samples)
        return np.argmax(final_output, axis=0)

    def calculate_multiclass_performance_metrics(self, y_true, y_pred):
        # y_true = np.array(y_true)
        # y_pred = np.array(y_pred)
        precision_per_class = []
        f1_per_class = []
        recall_per_class = []
        classes = np.unique(np.concatenate([y_true, y_pred]))
        for single_class in classes:
            TP = sum((y_pred == single_class) & (y_true == single_class))
            FP = sum((y_pred == single_class) & (y_true != single_class))
            FN = sum((y_pred != single_class) & (y_true == single_class))

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            precision_per_class.append(precision)
            f1_per_class.append(f1)
            recall_per_class.append(recall)

        macro_precision = np.mean(precision_per_class)
        macro_f1 = np.mean(f1_per_class)
        macro_recall = np.mean(recall_per_class)

        return macro_precision, macro_f1, macro_recall

    def calculate_recall(self, Y_true, Y_pred):
        TP, FP, FN, TN = self.confusion_matrix(Y_true, Y_pred)
        # print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        if (TP + FN > 0):
            recall_value = TP / float(TP + FN)
        else:
            recall_value = 0
        return recall_value

    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e_z / np.sum(e_z, axis=0, keepdims=True)

    def confusion_matrix(self, Y_true, Y_pred):
        TP = 0
        FP = 0
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

    # method to calculate prediction
    def calculate_precision(self, Y_true, Y_pred):
        TP, FP, FN, TN = self.confusion_matrix(Y_true, Y_pred)
        if (TP + FP > 0):
            precision_value = TP / float(TP + FP)
        else:
            precision_value = 0
        return precision_value

    def calculate_f1(self, Y_true, Y_pred):
        beta = 1
        F1 = 0
        TP, FP, FN, TN = self.confusion_matrix(Y_true, Y_pred)
        precision_value = self.calculate_precision(Y_true, Y_pred)
        recall_value = self.calculate_recall(Y_true, Y_pred)
        if (precision_value + recall_value > 0):
            # value = 1+(beta**2)
            # denominator = ((beta**2 * precision_value) + recall_value)
            # numerator = (1+(beta**2))*(precision_value * recall_value)
            # f1 = numerator/denominator

            F1 = 2 * precision_value * recall_value / (
                    precision_value + recall_value)  # for same weights for precision and recall
        return F1

def plot_learning_digital_digits(X_train, Y_train, X_test, Y_test, best_layers, best_lambd, learning_rate, step_size=10, epochs=300):

    test_costs = []
    train_sizes = list(range(5, X_train.shape[1] + 1, step_size))
    for size in train_sizes:
        Y_sub = Y_train[:, :size]
        X_sub = X_train[:, :size]
        nn = NeuralNetwork(layer_sizes=best_layers, lambd=best_lambd)
        for _ in range(epochs):
            _, activations = nn.forward_pass(X_sub)
            grads = nn.backward_pass(activations, X_sub, Y_sub)
            nn.update_parameters(grads, learning_rate)
        _, test_activations = nn.forward_pass(X_test)
        test_cost = nn.compute_cost(test_activations[-1], Y_test)
        test_costs.append(test_cost)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, test_costs, marker='o', linestyle='-', linewidth=2)
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Test Set Cost J")
    plt.title("Learning Curve (Digitial digit Dataset)")
    plt.grid(True)
    plt.show()

def load_data_sets():
    # Load digits data
    digits = datasets.load_digits(return_X_y=True)
    X, Y = digits
    return X, Y

if __name__ == "__main__":

    X_original, Y_original = load_data_sets()
    Y = OneHotEncoder(sparse_output=False, categories='auto').fit_transform(Y_original.reshape(-1, 1)).T

    df = pd.DataFrame(X_original)
    df['label'] = Y_original

    index = np.random.choice(len(df))
    plt.imshow(df.iloc[index, :-1].values.reshape(8, 8), cmap='gray')
    plt.title(f'Label: {df.iloc[index]["label"]}')
    plt.axis('off')
    plt.show()

    attributes = df.columns[:-1].tolist()

    neurons_list = [2, 4, 8, 16]
    layer_counts = [1, 2, 4, 8]
    lambdas = [0, 0.01, 0.1]
    results = []
    X = X_original.T
    input_size = X.shape[0]
    output_size = Y.shape[0]

    for num_layers in layer_counts:

        for neurons in neurons_list:
            for lambd in lambdas:
                layers = [neurons] * num_layers
                nn = NeuralNetwork(layer_sizes=[input_size] + layers + [output_size], lambd=lambd)
                start_time = time.time()
                avg_accuracy, avg_f1_score = nn.cross_validate(X, Y, folds=10)
                elapsed = time.time() - start_time

                print(
                    f"Layers: {num_layers}, Neurons: {neurons}, Lambda: {lambd}, Accuracy: {avg_accuracy:.4f}, F1_score: {avg_f1_score: .4f}, Time: {elapsed:.2f}s")
                results.append((num_layers, neurons, lambd, avg_accuracy, avg_f1_score, elapsed))

    indices = np.arange(X.shape[1])
    np.random.shuffle(indices)
    X = X_original.T[:, indices]
    Y = Y[:, indices]
    split_idx = int(0.8 * X.shape[1])

    Y_train, Y_test = Y[:, :split_idx], Y[:, split_idx:]
    X_train, X_test = X[:, :split_idx], X[:, split_idx:]
    best_lambda = 0.01
    learning_rate = 0.01
    # learning curve, this value is for credit approval data sets
    best_layers = [X.shape[0], 16, 16, 10]
    plot_learning_digital_digits(X_train, Y_train, X_test, Y_test, best_layers, best_lambda, learning_rate)