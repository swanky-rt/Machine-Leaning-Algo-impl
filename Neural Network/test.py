import numpy as np
import pandas as pd


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
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward_pass(self, X):
        activations = []
        pre_activations = []

        A = X  # shape: (m, n)
        for W in self.weights:
            A = np.insert(A, 0, 1, axis=1)
            print(f"A.shape: {A.shape}, W.T.shape: {W.T.shape}")# Add bias unit: shape (m, n+1)
            Z = A @ W.T                    # shape: (m, neurons)
            pre_activations.append(Z)
            A = self.sigmoid(Z)
            activations.append(A)

        return pre_activations, activations

    def compute_cost(self, output, Y):
        m = Y.shape[0]
        epsilon = 1e-5
        J = -np.sum(Y * np.log(output + epsilon) + (1 - Y) * np.log(1 - output + epsilon)) / m

        # Add regularization (exclude bias weights)
        for W in self.weights:
            J += (self.lambd / (2 * m)) * np.sum(W[:, 1:] ** 2)

        return J

    def backward_pass(self, activations, Y, X):
        m = Y.shape[0]
        grads = [np.zeros_like(W) for W in self.weights]
        deltas = [None] * (self.num_layers - 1)

        A = [X] + activations
        for i in range(len(A) - 1):
            A[i] = np.insert(A[i], 0, 1, axis=1)  # Add bias column

        deltas[-1] = activations[-1] - Y  # output layer delta
        for l in reversed(range(self.num_layers - 2)):
            W_next = self.weights[l + 1][:, 1:]  # remove bias column from next layer's weights
            delta_next = deltas[l + 1]
            derivative = self.sigmoid_derivative(A[l + 1][:, 1:])  # skip bias in current layer's activation
            deltas[l] = delta_next @ W_next * derivative

        for l in range(len(grads)):
            grads[l] = (deltas[l].T @ A[l]) / m
            grads[l][:, 1:] += (self.lambd / m) * self.weights[l][:, 1:]

        return grads

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

# Example usage:
def run_backprop_example1():
    print("Running backprop_example1.txt verification...")
    nn = NeuralNetwork(layer_sizes=[1, 2, 1], lambd=0.0)

    # Set weights from benchmark
    nn.weights[0] = np.array([[0.4, 0.1], [0.3, 0.2]])
    nn.weights[1] = np.array([[0.7, 0.5, 0.6]])

    X = np.array([[0.13], [0.42]])  # shape (2, 1)
    Y = np.array([[0.9], [0.23]])

    _, activations = nn.forward_pass(X)
    J = nn.compute_cost(activations[-1], Y)
    print(f"Final cost J: {J:.5f}")

    grads = nn.backward_pass(activations, Y, X)
    for i, g in enumerate(grads):
        print(f"Gradients for Theta{i + 1}:\n", np.round(g, 5))

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

    X = np.array([[0.32, 0.83], [0.68, 0.02]])  # shape (2, 2)
    Y = np.array([[0.75, 0.75], [0.98, 0.28]])  # shape (2, 2)

    _, activations = nn.forward_pass(X)
    J = nn.compute_cost(activations[-1], Y)
    print(f"Final cost J: {J:.5f}")

    grads = nn.backward_pass(activations, Y, X)
    for i, g in enumerate(grads):
        print(f"Gradients for Theta{i + 1}:\n", np.round(g, 5))

if __name__ == "__main__":
    run_backprop_example1()
    run_backprop_example2()
