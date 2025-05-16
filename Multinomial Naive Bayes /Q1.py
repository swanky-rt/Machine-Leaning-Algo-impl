from Utils import load_training_set, load_test_set, calculate_precision, calculate_recall, confusion_matrix, decide_class_of_doc
from Utils import prior_probability, plot_heat_map

def class_probability(vocab, values, class_name):
    word_class_probability = {}
    word_frequency = {}
    for word in vocab:
        word_frequency[word] = 0
    total_word_count = 0

    for value in values:
        for word in value:
            if word in vocab:
                word_frequency[word] += 1
                total_word_count += 1

    for word in vocab:
        if total_word_count > 0:
            word_class_probability[word] = word_frequency[word] / float(total_word_count)
        else:
            word_class_probability[word] = 0

    return word_class_probability

'For Q1, executing it without laplace smoothing'

if __name__ == '__main__':
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2

    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    pr_pos, pr_neg = prior_probability(pos_test, neg_test)
    positive_word_probility = class_probability(vocab, pos_test, 'positive')
    negative_word_probility = class_probability(vocab, neg_test, 'negative')
    Y_True = []
    Y_Pred = []
    correct_predictions = 0
    total_correct_predictions = 0
    for doc in pos_test:
        Y_True.append(1)
        prediction = decide_class_of_doc(doc,positive_word_probility, negative_word_probility, pr_pos, pr_neg, vocab)
        if prediction == 'positive':
            Y_Pred.append(1)
            correct_predictions += 1
        else:
            Y_Pred.append(0)
        total_correct_predictions += 1

    for doc in neg_test:
        Y_True.append(0)
        prediction = decide_class_of_doc(doc,positive_word_probility, negative_word_probility, pr_pos, pr_neg, vocab)
        if prediction == 'negative':
            Y_Pred.append(0)
            correct_predictions += 1
        else:
            Y_Pred.append(1)
        total_correct_predictions += 1
    accuracy = correct_predictions / float(total_correct_predictions)
    accuracy = accuracy * 100
    print("Accuracy:", accuracy)
    precision = calculate_precision(Y_True, Y_Pred)
    recall = calculate_recall(Y_True, Y_Pred)
    recall = recall * 100
    precision = precision * 100
    TP, FP, FN, TN = confusion_matrix(Y_True, Y_Pred)
    print(f" Confusion Matrix [[TP, FN],[FP, TN]]: [[{TP},{FN}],[{FP},{TN}]]")

    print("Precision:", precision)
    print("Recall:", recall)

    plot_heat_map(Y_True, Y_Pred)