import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


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

    def sigmoid(self, z):
        z = np.array(z, dtype=np.float64)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward_pass(self, X):
        activations = []
        pre_activations = []
        A = X
        for W in self.weights:
            A = np.vstack([np.ones((1, A.shape[1])), A])
            Z = np.dot(W, A)
            A = self.sigmoid(Z)
            activations.append(A)
            pre_activations.append(Z)
        return pre_activations, activations

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
        if self.lambd > 0:
            reg_term = sum(np.sum(W[:, 1:] ** 2) for W in self.weights)
            cost += (self.lambd / (2 * m)) * reg_term
        return cost

    def backward_pass(self, activations, X, Y):
        m = Y.shape[1]
        grads = [np.zeros_like(W) for W in self.weights]
        deltas = [None] * (self.num_layers - 1)
        A_layers = [X] + activations[:-1]
        deltas[-1] = activations[-1] - Y
        for l in reversed(range(self.num_layers - 2)):
            W = self.weights[l + 1][:, 1:]
            delta = np.dot(W.T, deltas[l + 1]) * self.sigmoid_derivative(activations[l])
            deltas[l] = delta
        for l in range(self.num_layers - 1):
            A_prev = A_layers[l]
            A_prev = np.vstack([np.ones((1, A_prev.shape[1])), A_prev])
            grads[l] = (1 / m) * np.dot(deltas[l], A_prev.T)
            reg = np.copy(self.weights[l])
            reg[:, 0] = 0
            grads[l] += (self.lambd / m) * reg
        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(self.num_layers - 1):
            self.weights[l] -= learning_rate * grads[l]

    def train(self, X, Y, learning_rate=0.1, epochs=2000, epsilon=1e-6, verbose=True):
        previous_cost = float('inf')
        for epoch in range(epochs):
            _, activations = self.forward_pass(X)
            grads = self.backward_pass(activations, X, Y)
            cost = self.compute_cost(activations[-1], Y)
            self.update_parameters(grads, learning_rate)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost}")
            if abs(previous_cost - cost) < epsilon:
                if verbose:
                    print(f"Using stopping criteria: Stopping early at epoch {epoch} with cost {cost}")
                break
            previous_cost = cost

    def cross_validate(self, X, Y, folds=5, learning_rate=0.1, epsilon=1e-6, epochs=1000):
        f1_scores = []
        accuracies = []
        Y_labels = np.argmax(Y, axis=0) if Y.shape[0] > 1 else Y.flatten()
        classes = np.unique(Y_labels)
        fold_indices = [[] for _ in range(folds)]
        for c in classes:
            class_indices = np.where(Y_labels == c)[0]
            np.random.shuffle(class_indices)
            split = np.array_split(class_indices, folds)
            for i in range(folds):
                fold_indices[i].extend(split[i])
        for fold in range(folds):
            train_idx = [idx for i in range(folds) if i != fold for idx in fold_indices[i]]
            test_idx = fold_indices[fold]
            Y_train, Y_test = Y[:, train_idx], Y[:, test_idx]
            X_train, X_test = X[:, train_idx], X[:, test_idx]
            self.__init__(self.layer_sizes, lambd=self.lambd)
            self.train(X_train, Y_train, learning_rate=learning_rate, epochs=epochs, epsilon=epsilon, verbose=False)
            labels = Y_test.flatten()
            predictions = self.predict(X_test)
            accuracy = np.mean(predictions == labels)
            f1_score = self.calculate_f1(predictions, labels)
            accuracies.append(accuracy)
            f1_scores.append(f1_score)
        return np.mean(accuracies), np.mean(f1_scores)

    def predict(self, X):
        _, activations = self.forward_pass(X)
        output = activations[-1]
        return (output > 0.5).astype(int).flatten()

    def calculate_recall(self, Y_true, Y_pred):
        TP, FP, FN, TN = self.confusion_matrix(Y_true, Y_pred)
        # print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        if (TP + FN > 0):
            recall_value = TP / float(TP + FN)
        else:
            recall_value = 0
        return recall_value

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

def run_backprop_example1():
    print("Running backprop_example1.txt verification...")
    nn = NeuralNetwork(layer_sizes=[1, 2, 1], lambd=0.0)
    nn.weights[0] = np.array([[0.4, 0.1], [0.3, 0.2]])
    nn.weights[1] = np.array([[0.7, 0.5, 0.6]])
    X = np.array([[0.13, 0.42]])  # shape (1, 2)
    Y = np.array([[0.9, 0.23]])
    _, activations = nn.forward_pass(X)
    J = nn.compute_cost(activations[-1], Y)
    print(f"Final cost J: {J:.5f}")
    grads = nn.backward_pass(activations, X, Y)
    for i, g in enumerate(grads):
        print(f"Gradients for Theta{i + 1}:")
        print(np.round(g, 5))

def run_backprop_example2():
    print("\nRunning backprop_example2.txt verification...")
    nn = NeuralNetwork(layer_sizes=[2, 4, 3, 2], lambd=0.25)
    nn.weights[0] = np.array([
        [0.42, 0.15, 0.40],
        [0.72, 0.10, 0.54],
        [0.01, 0.19, 0.42],
        [0.30, 0.35, 0.68]
    ])
    nn.weights[1] = np.array([
        [0.21, 0.67, 0.14, 0.96, 0.87],
        [0.87, 0.42, 0.20, 0.32, 0.89],
        [0.03, 0.56, 0.80, 0.69, 0.09]
    ])
    nn.weights[2] = np.array([
        [0.04, 0.87, 0.42, 0.53],
        [0.17, 0.10, 0.95, 0.69]
    ])
    X = np.array([[0.32, 0.83], [0.68, 0.02]])
    Y = np.array([[0.75, 0.75], [0.98, 0.28]])
    _, activations = nn.forward_pass(X)
    J = nn.compute_cost(activations[-1], Y)
    print(f"Final cost J: {J:.5f}")
    grads = nn.backward_pass(activations, X, Y)
    for i, g in enumerate(grads):
        print(f"Gradients for Theta{i + 1}:")
        print(np.round(g, 5))

def plot_learning_curve_loan(X_train, Y_train, X_test, Y_test, best_layers, best_lambd, learning_rate, step_size=10, epochs=500):
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
    plt.title("Learning Curve (Loan Dataset)")
    plt.grid(True)
    plt.show()

def load_loan_dataset(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(index=0).reset_index(drop=True)
    X = data.drop(columns=['label'])
    Y = data['label'].values
    categorical_cols = [col for col in X.columns if 'cat' in col]
    numerical_cols = [col for col in X.columns if 'num' in col]
    X_encoded = pd.get_dummies(X, columns=categorical_cols)
    X_encoded = X_encoded.astype(np.float64)
    for col in numerical_cols:
        col_max = X_encoded[col].max()
        col_min = X_encoded[col].min()
        if col_max != col_min:
            X_encoded[col] = 2 * (X_encoded[col] - col_min) / (col_max - col_min) - 1
        else:
            X_encoded[col] = 0
    X = X_encoded.values.T
    Y = Y.reshape(1, -1)
    return X, Y

if __name__ == "__main__":
    run_backprop_example1()
    run_backprop_example2()

    filename = 'loan.csv'
    filepath = '/../../Users/amitkumar/Documents/ML/HW4/datasets/'
    X, Y = load_loan_dataset(filepath+filename)

    neurons_list = [2, 4, 8, 16]
    layer_counts = [1, 2, 4, 8]
    lambdas = [0, 0.01, 0.1]
    results = []

    for num_layers in layer_counts:

        for neurons in neurons_list:
            for lambd in lambdas:
                layers = [neurons] * num_layers
                nn = NeuralNetwork(layer_sizes=[X.shape[0]] + layers + [1], lambd=lambd)
                start_time = time.time()
                avg_accuracy, avg_f1_score = nn.cross_validate(X, Y, folds=5)
                elapsed = time.time() - start_time

                print(
                    f"Layers: {num_layers}, Neurons: {neurons}, Lambda: {lambd}, Accuracy: {avg_accuracy:.4f}, F1_score: {avg_f1_score: .4f}, Time: {elapsed:.2f}s")
                results.append((num_layers, neurons, lambd, avg_accuracy, avg_f1_score, elapsed))

    split_idx = int(0.8 * X.shape[1])

    Y_train, Y_test = Y[:, :split_idx], Y[:, split_idx:]
    X_train, X_test = X[:, :split_idx], X[:, split_idx:]
    best_lambda = 0.01
    learning_rate = 0.01
    # learning curve, this value is for loan data sets
    best_layers = list(map(int, [X.shape[0], 8, 1]))  # 1 hidden layer with 8 neurons
    plot_learning_curve_loan(X_train, Y_train, X_test, Y_test, best_layers, best_lambda, learning_rate)
    print("its printing")