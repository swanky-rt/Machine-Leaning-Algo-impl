import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt
from sklearn.metrics import f1_score


class NeuralNetwork:
    def __init__(self, layer_sizes, seed=42, lambd=0.0):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.lambd = lambd

        self.weights = []
        for i in range(self.num_layers - 1):
            # Add +1 for bias in input to current layer
            weight_matrix = np.random.randn(layer_sizes[i+1], layer_sizes[i] + 1) * 0.01
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
            A = np.vstack([np.ones((1, A.shape[1])), A])  # Add bias unit
            Z = np.dot(W, A)
            A = self.sigmoid(Z)
            pre_activations.append(Z)
            activations.append(A)

        return pre_activations, activations

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))

        if self.lambd > 0:
            reg_term = 0
            for W in self.weights:
                reg_term += np.sum(np.square(W[:, 1:]))  # exclude bias column
            cost += (self.lambd / (2 * m)) * reg_term

        return cost

    def backward_pass(self, activations, Y, X):
        m = Y.shape[1]
        deltas = [None] * (self.num_layers - 1)
        grads = [np.zeros_like(W) for W in self.weights]

        # Append input layer activation
        A_layers = [X]
        for A in activations[:-1]:
            A_layers.append(A)

        # Output layer error
        delta = activations[-1] - Y
        deltas[-1] = delta

        for l in reversed(range(self.num_layers - 2)):
            W = self.weights[l + 1][:, 1:]  # Remove bias from next layer weights
            delta = np.dot(W.T, deltas[l + 1]) * self.sigmoid_derivative(activations[l])
            deltas[l] = delta

        for l in range(self.num_layers - 1):
            A_prev = A_layers[l]
            A_prev = np.vstack([np.ones((1, A_prev.shape[1])), A_prev])  # Add bias
            grads[l] = (1/m) * np.dot(deltas[l], A_prev.T)

            # Add regularization (excluding bias)
            reg = np.copy(self.weights[l])
            reg[:, 0] = 0  # don't regularize bias
            grads[l] += (self.lambd / m) * reg

        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(self.num_layers - 1):
            self.weights[l] -= learning_rate * grads[l]

    def debug_backprop(self, X, Y):
        print("\n=== Forward Pass ===")
        Zs, activations = self.forward_pass(X)

        for idx, (z, a) in enumerate(zip(Zs, activations)):
            print(f"Layer {idx + 1} pre-activation (Z):\n{z}")
            print(f"Layer {idx + 1} activation (A):\n{a}")

        print(f"\nFinal Prediction (Output Layer):\n{activations[-1]}")
        print(f"Expected Output:\n{Y}")

        cost = self.compute_cost(activations[-1], Y)
        print(f"Cost J: {cost}\n")

        print("=== Backward Pass ===")
        grads = self.backward_pass(activations, Y, X)
        for l, grad in enumerate(grads):
            print(f"Gradient for Layer {l + 1}:\n{grad}")

    def train(self, X, Y, learning_rate=0.01, epochs=1000, epsilon=1e-6, verbose=True):
        previous_cost = float('inf')
        for epoch in range(epochs):
            _, activations = self.forward_pass(X)
            cost = self.compute_cost(activations[-1], Y)
            grads = self.backward_pass(activations, Y, X)
            self.update_parameters(grads, learning_rate)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost}")
            #Early Stopping Based on Convergence of Cost (J)( second stopping criteria)
            if abs(previous_cost - cost) < epsilon:
                if verbose:
                    print(f"Stopping early at epoch {epoch} with cost {cost}")
                break
            previous_cost = cost

    def calculate_recall(self, Y_true, Y_pred):
        TP, FP, FN, TN = self.confusion_matrix(Y_true, Y_pred)
        print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
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

    def cross_validate(self, X, Y, folds=5, learning_rate=0.1, epochs=1000, epsilon=1e-6):
        accuracies = []
        f1_scores = []
        Y_labels = np.argmax(Y, axis=0) if Y.shape[0] > 1 else Y.flatten()

        classes = np.unique(Y_labels)
        class_indices = {c: np.where(Y_labels == c)[0] for c in classes}

        for c in class_indices:
            np.random.shuffle(class_indices[c])

        folds_indices = [[] for _ in range(folds)]
        for c in class_indices:
            split_indices = np.array_split(class_indices[c], folds)
            for i in range(folds):
                folds_indices[i].extend(split_indices[i])

        for fold_index in range(folds):
            test_idx = folds_indices[fold_index]
            train_idx = [idx for i, fold in enumerate(folds_indices) if i != fold_index for idx in fold]

            X_train, X_test = X[:, train_idx], X[:, test_idx]
            Y_train, Y_test = Y[:, train_idx], Y[:, test_idx]

            self.__init__(self.layer_sizes, lambd=self.lambd)
            self.train(X_train, Y_train, learning_rate=learning_rate, epochs=epochs, epsilon=epsilon, verbose=False)

            predictions = self.predict(X_test)
            labels = Y_test.flatten()
            accuracy = np.mean(predictions == labels)
            accuracies.append(accuracy)
            f1_score = self.calculate_f1(predictions, labels)
            f1_scores.append(f1_score)

        return np.mean(accuracies), np.mean(f1_scores)

    def predict(self, X):
        _, activations = self.forward_pass(X)
        output = activations[-1]
        return (output > 0.5).astype(int).flatten()

def plot_learning_curve(X_train, Y_train, X_test, Y_test, best_layers, best_lambd, learning_rate=0.1, step_size=5):
    m = X_train.shape[1]
    sample_sizes = list(range(step_size, m + 1, step_size))
    costs = []

    for num_samples in sample_sizes:
        X_subset = X_train[:, :num_samples]
        Y_subset = Y_train[:, :num_samples]

        nn = NeuralNetwork(layer_sizes=best_layers, lambd=best_lambd)
        nn.train(X_subset, Y_subset, learning_rate=learning_rate, epochs=1000, epsilon=1e-6, verbose=False)

        _, activations_test = nn.forward_pass(X_test)
        test_cost = nn.compute_cost(activations_test[-1], Y_test)
        costs.append(test_cost)

    plt.figure(figsize=(8, 6))
    plt.plot(sample_sizes, costs, marker='o')
    plt.title('Learning Curve on Loan Dataset')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Test Set Cost J')
    plt.grid(True)
    plt.show()

def load_loan_dataset(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(index=0).reset_index(drop=True)
    X = data.drop(columns=['label'])
    Y = data['label'].values

    categorical_cols = [col for col in X.columns if 'cat' in col]
    numerical_cols = [col for col in X.columns if 'num' in col]

    X_encoded = pd.get_dummies(X, columns=categorical_cols) #hot encoding on categories
    X_encoded = X_encoded.astype(np.float64)

    for col in numerical_cols:
        col_max = X_encoded[col].max()
        col_min = X_encoded[col].min()
        if col_max != col_min:
            X_encoded[col] = 2 * (X_encoded[col] - col_min) / (col_max - col_min) - 1
        else:
            X_encoded[col] = 0

    X = X_encoded.values
    X = X.T
    Y = Y.reshape(1, -1)
    return X, Y

def load_wdbc_dataset(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(index=0).reset_index(drop=True)
    X = data.drop(columns=['label'])
    Y = data['label'].values

    categorical_cols = [col for col in X.columns if 'cat' in col]
    numerical_cols = [col for col in X.columns if 'num' in col]

    X_encoded = pd.get_dummies(X, columns=categorical_cols) # hot encoding
    X_encoded = X_encoded.astype(np.float64)

    for col in numerical_cols:
        col_max = X_encoded[col].max()
        col_min = X_encoded[col].min()
        if col_max != col_min:
            X_encoded[col] = 2 * (X_encoded[col] - col_min) / (col_max - col_min) - 1
        else:
            X_encoded[col] = 0

    X = X_encoded.values
    X = X.T
    Y = Y.reshape(1, -1)
    return X, Y

def run_backprop_example1():
    print("Running backprop_example1.txt verification...")

    nn = NeuralNetwork(layer_sizes=[1, 2, 1], lambd=0.0)
    # Initialize weights and biases (bias included in weights)
    nn.weights[0] = np.array([[0.4, 0.1],
                              [0.3, 0.2]])
    nn.weights[1] = np.array([[0.7, 0.5, 0.6]])

    X = np.array([[0.13, 0.42]])  # shape (1, 2)
    Y = np.array([[0.9, 0.23]])   # shape (1, 2)

    nn.debug_backprop(X, Y)

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

    X = np.array([[0.32, 0.83],
                  [0.68, 0.02]])  # shape: (2, 2)
    Y = np.array([[0.75, 0.75],
                  [0.98, 0.28]])  # shape: (2, 2)

    nn.debug_backprop(X, Y)

    nn.debug_backprop(X, Y)
if __name__ == "__main__":
    run_backprop_example1()
    run_backprop_example2()

    #neural network
    filename = 'wdbc.csv'
    filepath = '/../../Users/amitkumar/Documents/ML/HW4/datasets/'
    filename1 = 'loan.csv'
    filename3 = 'raisin.csv'
    filename4 = 'titanic.csv'
    X, Y = load_loan_dataset(filepath+filename3)

    neurons_list = [2, 4, 8, 16]
    layer_counts = [1, 2, 3, 4]
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
                print(f"Layers: {num_layers}, Neurons: {neurons}, Lambda: {lambd}, Accuracy: {avg_accuracy:.4f}, F1_score: {avg_f1_score: .4f}, Time: {elapsed:.2f}s")
                results.append((num_layers, neurons, lambd, avg_accuracy, avg_f1_score, elapsed))

    X1, Y1 = load_loan_dataset(filepath+filename1)

    # Step 2: Split into 80% Train, 20% Test
    split_idx = int(0.8 * X1.shape[1])
    X_train, X_test = X1[:, :split_idx], X1[:, split_idx:]
    Y_train, Y_test = Y1[:, :split_idx], Y1[:, split_idx:]

    #learning curve, this value is for loan data sets
    # best_layers = [X1.shape[0], 2, 1]  # 1 hidden layer with 2 neurons
    # best_lambda = 0  # lambda=0
    # learning_rate = 0.1
    # step_size = 5

    # learning curve, this value is for wdbc data sets
    best_layers = [X1.shape[0], 2, 1]  # 1 hidden layer with 2 neurons
    best_lambda = 0.1  # lambda=0
    learning_rate = 0.1
    step_size = 5



    #plot_learning_curve(X_train, Y_train, X_test, Y_test, best_layers, best_lambda, learning_rate, step_size)

