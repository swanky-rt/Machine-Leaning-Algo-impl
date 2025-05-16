import numpy as np
import random
from matplotlib import pyplot as plt


from Utils import load_training_set, load_test_set, calculate_precision, calculate_recall, confusion_matrix
from Utils import plot_heat_map

def class_probability_with_laplace_soothing(vocab, values, class_name, alpha=1):
    word_class_probability = {}
    total_word_count = 0
    word_frequency = {}
    for word in vocab:
        word_frequency[word] = 0

    for value in values:
        for word in value:
            if word in vocab:
                word_frequency[word] += 1
                total_word_count += 1

    for word in vocab:
            word_class_probability[word] = (word_frequency[word] + alpha)/ (total_word_count + alpha * len(vocab))

    return word_class_probability

def plot_graph(alpha_values, accuracies):
    plt.figure(figsize=(8, 6))
    plt.plot(alpha_values, accuracies, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.xlabel('alpha α (log scale')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs laplace smoothing')
    plt.grid(True)
    plt.show()

def log_prior_probability(positive_word_prob, pr_pos, pr_neg, doc, vocab, negative_word_prob):
    log_pr_pos = np.log(pr_pos)
    log_pr_neg = np.log(pr_neg)
    for word in doc:
        if word in vocab:
            log_pr_pos += np.log(positive_word_prob.get(word, 1e-10))
            log_pr_neg += np.log(negative_word_prob.get(word, 1e-10))
    if log_pr_pos > log_pr_neg:
        return "positive"
    elif log_pr_pos < log_pr_neg:
        return "negative"
    else:
        return random.choices(["positive", "negative"])

def prior_probability(positive_values, negative_values):
    N = len(positive_values) + len(negative_values)
    N_positive = len(positive_values)
    N_negative = len(negative_values)
    Pr_pos = N_positive/float(N)
    Pr_neg = N_negative/float(N)
    return Pr_pos, Pr_neg

'For Q2, executing it for multiple alpha values with laplace smoothing'

if __name__ == '__main__':
    percentage_positive_instances_train = 0.3
    percentage_negative_instances_train = 0.3

    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    pr_pos, pr_neg = prior_probability(pos_test, neg_test)
    accuracies = []

    positive_word_probability = class_probability_with_laplace_soothing(vocab, pos_train, 'positive', 1)
    negative_word_probability = class_probability_with_laplace_soothing(vocab, neg_train, 'negative', 1)
    correct_predictions = 0
    total_correct_predictions = 0
    Y_True = []
    Y_Pred = []

    for doc in pos_test:
        Y_True.append(1)
        prediction = log_prior_probability(positive_word_probability, pr_pos, pr_neg, doc, vocab, negative_word_probability)
        if prediction == 'positive':
            Y_Pred.append(1)
            correct_predictions += 1
        else:
            Y_Pred.append(0)
        total_correct_predictions += 1

    for doc in neg_test:
        Y_True.append(0)
        prediction = log_prior_probability(positive_word_probability, pr_pos, pr_neg, doc, vocab, negative_word_probability)
        if prediction == 'negative':
            Y_Pred.append(0)
            correct_predictions += 1
        else:
            Y_Pred.append(1)
        total_correct_predictions += 1

    accuracy = correct_predictions / float(total_correct_predictions)

    accuracies.append(accuracy)
    print("Accuracy:", accuracy)

    precision = calculate_precision(Y_True, Y_Pred)
    recall = calculate_recall(Y_True, Y_Pred)
    TP, FP, FN, TN = confusion_matrix(Y_True, Y_Pred)
    print(f" Confusion Matrix [[TP, FN],[FP, TN]]: [[{TP},{FN}],[{FP},{TN}]]")

    recall = recall * 100
    precision = precision * 100
    print(f"Precision value is {precision}")
    print(f"recall value is {recall}")

    plot_heat_map(Y_True, Y_Pred)