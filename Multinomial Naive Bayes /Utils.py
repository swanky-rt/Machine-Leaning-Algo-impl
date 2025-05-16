import re
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import nltk
import random
import seaborn as sns
import numpy as np

REPLACE_NO_SPACE = re.compile("[._;:!*`¦\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
nltk.download('stopwords')
filename = '/../../Users/amitkumar/Documents/ML/HW2_CMPSCI_589_Spring2025_Supporting_Files/starter_code/'
train_positive = filename + 'train-positive.csv'
train_negative = filename + 'train-negative.csv'
test_positive = filename + 'test-positive.csv'
test_negative = filename + 'test-negative.csv'


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = REPLACE_NO_SPACE.sub("", text)
    text = REPLACE_WITH_SPACE.sub(" ", text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = text.split()
    return [w for w in words if w not in stop_words]


def load_training_set(percentage_positives, percentage_negatives):
    vocab = set()
    positive_instances = []
    negative_instances = []

    df = pd.read_csv(train_positive)
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_positives:
            continue
        contents = preprocess_text(contents)
        positive_instances.append(contents)
        vocab = vocab.union(set(contents))

    df = pd.read_csv(train_negative)
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_negatives:
            continue
        contents = preprocess_text(contents)
        negative_instances.append(contents)
        vocab = vocab.union(set(contents))

    return positive_instances, negative_instances, vocab


def load_test_set(percentage_positives, percentage_negatives):
    positive_instances = []
    negative_instances = []

    df = pd.read_csv(test_positive)
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_positives:
            continue
        contents = preprocess_text(contents)
        positive_instances.append(contents)

    df = pd.read_csv(test_negative)

    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_negatives:
            continue
        contents = preprocess_text(contents)
        negative_instances.append(contents)

    return positive_instances, negative_instances

#method to calculate prediction
def calculate_precision(Y_true, Y_pred):
    TP, FP, FN, TN = confusion_matrix(Y_true, Y_pred)
    if(TP + FP > 0):
        precision_value = TP/ float(TP+FP)
    else:
        precision_value = 0
    return precision_value

#method to calculate accuracy
def calculate_accuracy(Y_true, Y_pred):
    TP, FP, FN, TN = confusion_matrix(Y_true, Y_pred)
    total_value = float(TP+FP+FN+TN)
    return float(TP+TN)/total_value

def calculate_recall(Y_true, Y_pred):
    TP, FP, FN, TN = confusion_matrix(Y_true, Y_pred)
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

def decide_class_of_doc(doc, positive_word_prob, negative_word_prob, pr_pos, pr_neg, vocab):
    positive_probability = pr_pos
    negative_probability = pr_neg

    for word in doc:
        if word in vocab:
            positive_probability *= positive_word_prob.get(word, 0)
            negative_probability *= negative_word_prob.get(word, 0)
        else:
            positive_probability *= 1e-10
            negative_probability *= 1e-10

        if positive_probability > negative_probability:
            return ("positive")
        elif positive_probability < negative_probability:
            return ("negative")
        else:
            return random.choices(["positive", "negative"])[0]

def class_probability_with_laplace_soothing(vocab, values, class_name, alpha):
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

def class_probability_with_laplace_soothing_better(vocab, values, class_name, alpha, pr_pos, pr_neg):
    word_class_probability = {}
    word_frequency = {word: 0 for word in vocab}
    total_word_count = 0

    for value in values:
        for word in value:
            if word in vocab:
                word_frequency[word] += 1
                total_word_count += 1

    for word in vocab:
            word_class_probability[word] = (word_frequency[word] + alpha)/ (total_word_count + alpha * len(vocab))
    if class_name == 'positive':
        class_proabability = pr_pos
    else:
        class_proabability = pr_neg
    final_class_proabability = class_proabability
    for word in values:
        final_class_proabability *= word_class_probability.get(word, 1e-10)

    return final_class_proabability

def prior_probability(positive_values, negative_values):
    N = len(positive_values) + len(negative_values)
    N_positive = len(positive_values)
    N_negative = len(negative_values)
    Pr_pos = N_positive/float(N)
    Pr_neg = N_negative/float(N)
    return Pr_pos, Pr_neg

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
        return random.choices(["positive", "negative"])[0]

def plot_heat_map(Y_True, Y_pred):
    TP, FP, FN, TN = confusion_matrix(Y_True, Y_pred)
    conf_matrix = np.array([[TN, FP], [FN, TP]])
    plt.figure(figsize = (6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['predicted_negative', 'predicted_positive'],yticklabels=['true_negative', 'true_positive'], cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

def execute_data(data, batch):

    for i in range(0, len(data), batch):
        yield data[i:i + batch]

def execute_data_with_dicts(pos_test, neg_test, vocab, alpha, batch=100):
    p_pos, n_neg = prior_probability(pos_test, neg_test)


    positive_word_freq = {word: 0 for word in vocab}
    negative_word_freq = {word: 0 for word in vocab}

    for doc in execute_data(pos_test, batch):
        for d in doc:
            for word in d:
                if word in positive_word_freq:
                    positive_word_freq[word] += 1

    for doc in execute_data(neg_test, batch):
        for d in doc:
            for word in d:
                if word in negative_word_freq:
                    negative_word_freq[word] += 1


    positive_words_count = sum(positive_word_freq.values()) + len(vocab) * alpha
    negative_words_count = sum(negative_word_freq.values()) + len(vocab) * alpha

    for word in positive_word_freq:
        positive_word_freq[word] = (positive_word_freq[word] + alpha) / positive_words_count
    for word in negative_word_freq:
        negative_word_freq[word] = (negative_word_freq[word] + alpha) / negative_words_count

    Y_true = []
    Y_pred = []

    for doc in execute_data(pos_test, batch):
        for d in doc:
            Y_true.append(1)
            prediction = log_prior_probability(positive_word_freq, p_pos, n_neg, d, vocab, negative_word_freq)
            Y_pred.append(1 if prediction == 'positive' else 0)

    for doc in execute_data(neg_test, batch):
        for d in doc:
            Y_true.append(0)
            prediction = log_prior_probability(positive_word_freq, p_pos, n_neg, d, vocab, negative_word_freq)
            Y_pred.append(0 if prediction == 'negative' else 1)


    accuracy = calculate_accuracy(Y_true, Y_pred)
    print("Accuracy:", accuracy)
    precision = calculate_precision(Y_true, Y_pred)
    recall = calculate_recall(Y_true, Y_pred)
    TP, FP, FN, TN = confusion_matrix(Y_true, Y_pred)
    print(f" Confusion Matrix [[TP, FN],[FP, TN]]: [[{TP},{FN}],[{FP},{TN}]]")
    print(f"Precision value is {precision}")
    print(f"Recall value is {recall}")

    plot_heat_map(Y_true, Y_pred)
