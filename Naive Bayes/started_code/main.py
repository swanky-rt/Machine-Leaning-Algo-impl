from utils import load_training_set, load_test_set

if __name__ == '__main__':
    percentage_positive_instances_train = 0.1
    percentage_negative_instances_train = 0.1

    percentage_positive_instances_test = 0.1
    percentage_negative_instances_test = 0.1

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of positive training instances:", len(pos_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))
    