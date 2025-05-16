import numpy as np
import pandas as pd
import time

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
        z=np.clip(z,-500,500)
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
        # print("code is coming here in update parameter method")
        for l in range(self.num_layers - 1):
            self.weights[l] -= learning_rate * grads[l]


    def calculate_recall(self, Y_true, Y_pred):
        TP, FP, FN, TN = self.confusion_matrix(Y_true, Y_pred)
        #print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
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

            #self.__init__(self.layer_sizes, lambd=self.lambd)
            nn = NeuralNetwork(self.layer_sizes, lambd=self.lambd)
            nn.train(X_train, Y_train, lr=learning_rate, max_epochs=epochs, tol=epsilon, verbose=False)

            predictions = nn.predict(X_test)
            labels = Y_test.flatten()
            accuracies.append(np.mean(predictions == labels))
            f1_scores.append(nn.calculate_f1(predictions, labels))

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

def plot_learning_curve_parkinsons(X_train, Y_train, X_test, Y_test, best_layers, best_lambd, learning_rate, step_size=10, epochs=1000):
    import matplotlib.pyplot as plt

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
    plt.title("Learning Curve (Oxford's disease parkinsons Dataset)")
    plt.grid(True)
    plt.show()

def load_parkinsons_dataset(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(index=0).reset_index(drop=True)
    X = data.drop(columns=['Diagnosis'])
    Y = data['Diagnosis'].values

    numerical_cols = X.columns.tolist()  # All features are numerical in this dataset
    categorical_cols = []

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
    Y = (Y == Y.max()).astype(int).reshape(1, -1)
    return X, Y

if __name__ == "__main__":

    #neural network
    filename = 'parkinsons.csv'
    filepath = '/../../Users/amitkumar/Documents/ML/Final_Project_CMPSCI_589_Spring2025_Supporting_Files/'
    X, Y = load_parkinsons_dataset(filepath+filename)

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
                avg_accuracy, avg_f1_score = nn.cross_validate(X, Y, folds=10)
                elapsed = time.time() - start_time

                print(
                    f"Layers: {num_layers}, Neurons: {neurons}, Lambda: {lambd}, Accuracy: {avg_accuracy:.4f}, F1_score: {avg_f1_score: .4f}, Time: {elapsed:.2f}s")
                results.append((num_layers, neurons, lambd, avg_accuracy, avg_f1_score, elapsed))

    indices = np.arange(X.shape[1])
    np.random.shuffle(indices)
    X = X[:, indices]
    Y = Y[:, indices]
    split_idx = int(0.8 * X.shape[1])

    Y_train, Y_test = Y[:, :split_idx], Y[:, split_idx:]
    X_train, X_test = X[:, :split_idx], X[:, split_idx:]
    best_lambda = 0.01
    learning_rate = 0.01
    # learning curve, this value is for Oxford's disease parkinsons data sets
    best_layers = list(map(int, [X.shape[0], 16, 16, 1]))
    plot_learning_curve_parkinsons(X_train, Y_train, X_test, Y_test, best_layers, best_lambda, learning_rate)