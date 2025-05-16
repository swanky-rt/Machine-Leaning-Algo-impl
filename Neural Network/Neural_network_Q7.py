import numpy as np


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

    def numerical_gradient_check(self, X, Y, epsilon=1e-4):

        print(f"\nRunning numerical gradient checking with ε = {epsilon}")

        zs, activations = self.forward_pass(X)
        analytical_grads = self.backward_pass(activations, X, Y)

        numerical_grads = []
        for l, W in enumerate(self.weights):

            num_grad = np.zeros_like(W)
            for i in range(W.shape[0]):

                for j in range(W.shape[1]):
                    original_value = W[i, j]
                    self.weights[l][i, j] = original_value + epsilon
                    cost_plus = self.compute_cost(self.forward_pass(X)[1][-1], Y)
                    self.weights[l][i, j] = original_value - epsilon
                    cost_minus = self.compute_cost(self.forward_pass(X)[1][-1], Y)
                    num_grad[i, j] = (cost_plus - cost_minus) / (2 * epsilon)
                    self.weights[l][i, j] = original_value

            numerical_grads.append(num_grad)

        for l, (num, ana) in enumerate(zip(numerical_grads, analytical_grads)):

            print(f"\nGradient check for Theta{l + 1}:")

            for i in range(num.shape[0]):
                print("Analytical:", "  ".join(f"{val: .5f}" for val in ana[i]))
                print("Numerical :", "  ".join(f"{val: .5f}" for val in num[i]))

    def predict(self, X):
        _, activations = self.forward_pass(X)
        return (activations[-1] > 0.5).astype(int).flatten()

def run_benchmark_example1():

    print("Running backprop_example1.txt verification...")

    nn = NeuralNetwork(layer_sizes=[1, 2, 1], lambd=0.0)
    nn.weights[0] = np.array([[0.4, 0.1], [0.3, 0.2]])
    nn.weights[1] = np.array([[0.7, 0.5, 0.6]])

    X = np.array([[0.13, 0.42]])
    Y = np.array([[0.9, 0.23]])

    _, activations = nn.forward_pass(X)
    J = nn.compute_cost(activations[-1], Y)

    print(f"Final cost J: {J:.5f}")

    grads = nn.backward_pass(activations, X, Y)
    for i, g in enumerate(grads):

        print(f"Gradients for Theta{i + 1}:")
        print(np.round(g, 5))

    nn.numerical_gradient_check(X, Y, epsilon=0.1)
    nn.numerical_gradient_check(X, Y, epsilon=1e-6)

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

    _, activations = nn.forward_pass(X)
    J = nn.compute_cost(activations[-1], Y)

    print(f"Final cost J: {J:.5f}")

    grads = nn.backward_pass(activations, X, Y)
    for i, g in enumerate(grads):

        print(f"Gradients for Theta{i + 1}:")
        print(np.round(g, 5))

    nn.numerical_gradient_check(X, Y, epsilon=0.1)
    nn.numerical_gradient_check(X, Y, epsilon=1e-6)

if __name__ == "__main__":

    run_benchmark_example1()
    run_benchmark_example2()