from collections import Counter
import random
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as pltp


def generateWordFreq(doc_vector):
    return Counter(doc_vector).items()


def trainDataFormatter(labels, trainData):
    trainDataMap = {}
    for i in range(len(labels)):
        trainDataMap[labels[i]] = trainData[i]
    return trainDataMap


def extendList(*lists):
    finalList = []
    for list in lists:
        finalList.extend(list)
    return finalList


def randomColorGenerator():
    return plt.colors.to_hex((random.random(), random.random(), random.random()))
    # Get most and least frequent words
    N = 10  # Number of words to show from each end
    most_common = dict(Counter(word_freq).most_common(N))
    least_common = dict(sorted(Counter(word_freq).items(),
                               key=lambda x: x[1])[:N])

    # Create two subplots
    fig, (ax1, ax2) = pltp.subplots(2, 1, figsize=(15, 10))

    # Plot most frequent words
    words_most = list(most_common.keys())
    counts_most = np.array(list(most_common.values()))

    for alpha in alphas:
        probs_most = (counts_most + alpha) / (sum(counts_most) + alpha * vocabulary_size)
        ax1.plot(range(len(words_most)), probs_most,
                 marker='o', label=f"α = {alpha}")

    ax1.set_xticks(range(len(words_most)))
    ax1.set_xticklabels(words_most, rotation=45, ha='right')
    ax1.set_title("Most Frequent Words")
    ax1.set_ylabel("Probability")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot least frequent words
    words_least = list(least_common.keys())
    counts_least = np.array(list(least_common.values()))

    for alpha in alphas:
        probs_least = (counts_least + alpha) / (sum(counts_least) + alpha * vocabulary_size)
        ax2.plot(range(len(words_least)), probs_least,
                 marker='o', label=f"α = {alpha}")

    ax2.set_xticks(range(len(words_least)))
    ax2.set_xticklabels(words_least, rotation=45, ha='right')
    ax2.set_title("Least Frequent Words")
    ax2.set_xlabel("Words")
    ax2.set_ylabel("Probability")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    pltp.suptitle("Effect of Laplace Smoothing on Word Probabilities",
                  fontsize=14, y=1.02)
    pltp.tight_l