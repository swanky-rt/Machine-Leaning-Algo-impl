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

    def sigmoid_function(self, z):

        z = np.array(z, dtype=np.float64)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative_function(self, a):

        return a * (1 - a)

    def forward_pass(self, input_data):
        pre_acts = []
        acts = []
        current = input_data

        for w in self.weights:

            bias_row = np.ones((1, current.shape[1]))
            current_with_bias = np.vstack([bias_row, current])
            z = np.dot(w, current_with_bias)
            current = self.sigmoid_function(z)
            pre_acts.append(z)
            acts.append(current)

        return pre_acts, acts

    def compute_cost(self, output, target):
        num_samples = target.shape[1]
        term1 = target * np.log(output + 1e-8)
        term2 = (1 - target) * np.log(1 - output + 1e-8)
        base_cost = -(1 / num_samples) * np.sum(term1 + term2)

        if self.lambd > 0:
            reg_sum = 0
            for w in self.weights:
                reg_sum += np.sum(w[:, 1:] ** 2)
            reg_cost = (self.lambd / (2 * num_samples)) * reg_sum
            base_cost += reg_cost

        return base_cost

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

            predictions = self.predict(X_test)
            labels = Y_test.flatten()
            accuracies.append(np.mean(predictions == labels))
            f1_scores.append(self.calculate_f1(predictions, labels))

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
        final_output = acts[-1]
        binary = (final_output > 0.5).astype(int)
        return binary.flatten()

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

def run_benchmark_example1():
    print("Running backprop_example1.txt verification...")

    nn = NeuralNetwork(layer_sizes=[1, 2, 1], lambd=0.0)
    nn.weights[0] = np.array([[0.4, 0.1], [0.3, 0.2]])
    nn.weights[1] = np.array([[0.7, 0.5, 0.6]])


    X = np.array([[0.13, 0.42]])
    Y = np.array([[0.9, 0.23]])

    Zs, activations = nn.forward_pass(X)
    J = nn.compute_cost(activations[-1], Y)


    print(f"Final cost J: {J:.5f}")
    grads = nn.backward_pass(activations, X, Y)

    print("== Training instance 1 ==")

    print(f"Input x: [{X[0,0]}] Expected output y: [{Y[0,0]}]")
    print(f"Network Architecture: {nn.layer_sizes} Number of Layers: {nn.num_layers}")

    print("Forward Pass Values:")
    print(f"Layer 0 (Input with bias):")

    a0 = np.hstack(([1], X[0, 0]))
    print(f"Activations (a0): {a0}")
    print(f"Pre-activations (z1): {np.round(Zs[0][:,0], 6)}")
    print("Weights (Theta1):")

    print(nn.weights[0])
    print(f"Layer 1 (Input bias):")

    print(f"Activations (a1): {np.round(np.vstack([np.array([[1.0]]), activations[0][:, [0]]]).flatten(), 6)}")
    print(f"Pre-activations (z2): {np.round(Zs[1][:,0], 6)}")
    print(f"Weights (Theta2): {nn.weights[1]}")

    print(f"Output Layer (a2): {np.round(activations[-1][:,0], 6)}")
    instance1_cost = -(Y[0,0] * np.log(activations[-1][0,0]) + (1 - Y[0,0]) * np.log(1 - activations[-1][0,0]))

    print(f"Cost for instance 1: {instance1_cost:.6f}")

    print("== Training instance 2 ==")

    print(f"Input x: [{X[0,1]}] Expected output y: [{Y[0,1]}]")
    print(f"Network Architecture: {nn.layer_sizes} Number of Layers: {nn.num_layers}")

    print("Forward Pass Values:")
    print(f"Layer 0 (Input with bias):")
    a0 = np.hstack(([1], X[0, 1]))

    print(f"Activations (a0): {a0}")
    print(f"Pre-activations (z1): {np.round(Zs[0][:,1], 6)}")
    print("Weights (Theta1):")

    print(nn.weights[0])
    print(f"Layer 1 (Input bias):")

    print(f"Activations (a1): {np.round(np.vstack([np.array([[1.0]]), activations[0][:, [1]]]).flatten(), 6)}")
    print(f"Pre-activations (z2): {np.round(Zs[1][:,1], 6)}")
    print(f"Weights (Theta2): {nn.weights[1]}")

    print(f"Output Layer (a2): {np.round(activations[-1][:,1], 6)}")
    instance2_cost = -(Y[0,1] * np.log(activations[-1][0,1]) + (1 - Y[0,1]) * np.log(1 - activations[-1][0,1]))
    print(f"Cost for instance 2: {instance2_cost:.6f}")

    print("== Backpropagation Gradients ==")

    for i, g in enumerate(grads):

        print(f"Layer {i+1} Gradients (Theta{i+1}):")
        print(f"Dimensions: {g.shape}")

        print(np.round(g, 6))
        print(f"Weight Updates (alpha * gradient):")

        print(np.round(0.01 * g, 6))  # assuming alpha = 0.01


def run_benchmark_example2():
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

    Zs, activations = nn.forward_pass(X)
    J = nn.compute_cost(activations[-1], Y)
    print(f"Final cost J: {J:.5f}")

    for i in range(X.shape[1]):
        print(f"\n==Training instance {i+1} ==")
        print(f"Input x: {X[:,i]} Expected output y: {Y[:,i]}")

        print(f"Network Architecture: {nn.layer_sizes} Number of Layers: {nn.num_layers}")
        print("Forward Pass Values:")

        # Input layer
        print(f"Layer 0 (Input with bias):")

        a0 = np.hstack(([1], X[:, i]))
        print(f"Activations (a0): {np.round(a0, 6)}")

        print(f"Pre-activations (z1): {np.round(Zs[0][:, i], 6)}")
        print("Weights (Theta1):")

        print(nn.weights[0])

        print(f"Layer 1 (Input bias):")

        a1 = np.vstack([np.array([[1.0]]), activations[0][:, [i]]])
        print(f"Activations (a1): {np.round(a1.flatten(), 6)}")

        print(f"Pre-activations (z2): {np.round(Zs[1][:, i], 6)}")
        print("Weights (Theta2):")

        print(nn.weights[1])

        print(f"Layer 2 (Input bias):")
        a2 = np.vstack([np.array([[1.0]]), activations[1][:, [i]]])
        print(f"Activations (a2): {np.round(a2.flatten(), 6)}")

        print(f"Pre-activations (z3): {np.round(Zs[2][:, i], 6)}")

        print("Weights (Theta3):")
        print(nn.weights[2])

        print(f"Output Layer (a3): {np.round(activations[-1][:, i], 6)}")

        instance_cost = -np.sum(Y[:, i] * np.log(activations[-1][:, i] + 1e-8) +
                                (1 - Y[:, i]) * np.log(1 - activations[-1][:, i] + 1e-8))

        print(f"Cost for instance {i+1}: {instance_cost:.6f}")

    # Backpropagation Gradient Details
    print("\n== Backpropagation Gradients ==")

    grads = nn.backward_pass(activations, X, Y)

    for i, g in enumerate(grads):

        print(f"Layer {i+1} Gradients (Theta{i+1}):")
        print(f"Dimensions: {g.shape}")
        print(np.round(g, 6))
        print("Weight Updates (alpha * gradient):")

        print(np.round(0.01 * g, 6))  # assuming learning rate alpha = 0.01

def run_backprop_example1_without_activation():

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

def run_backprop_example2_without_activation():

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

    numerical_cols = [col for col in X.columns if 'num' in col]
    categorical_cols = [col for col in X.columns if 'cat' in col]

    X_encoded = pd.get_dummies(X, columns=categorical_cols) # hot encoding
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

    run_benchmark_example1()
    run_benchmark_example2()
    #run_backprop_example1_without_activation()
    #run_backprop_example2_without_activation()

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
    best_lambda = 0.1
    learning_rate = 0.01
    # learning curve, this value is for loan data sets
    best_layers = list(map(int, [X.shape[0], 16, 1]))
    plot_learning_curve_loan(X_train, Y_train, X_test, Y_test, best_layers, best_lambda, learning_rate)
    print("checking the plot")