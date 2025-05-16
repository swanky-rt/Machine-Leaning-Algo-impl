import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CalculateAccuracy:
    def __init__(self, test, pred, labels):
        self.test = test
        self.pred = pred
        self.labels = labels

    def accuracy(self):
        correct_predictions = np.sum(np.array(self.test) == np.array(self.pred))
        total_predictions = len(self.test)
        accuracy = correct_predictions / total_predictions
        return accuracy * 100
    
    def confusion_matrix(self):
        self.cm = np.zeros((len(self.labels), len(self.labels)), dtype=int)
        for i in range(len(self.test)):
            actual = self.test[i]
            predicted = self.pred[i]
            self.cm[self.labels.index(actual)][self.labels.index(predicted)] += 1
        return self.cm
    
    def precision(self):
        self.precision = np.zeros(len(self.labels))
        for i in range(len(self.labels)):
            n = self.cm[i][i]
            d = np.sum(self.cm[:,i])
            self.precision[i] = n / d
        return np.mean(self.precision)*100
    
    def recall(self):
        self.recall = np.zeros(len(self.labels))
        for i in range(len(self.labels)):
            n = self.cm[i][i]
            d = np.sum(self.cm[i,:])
            self.recall[i] = n / d
        return np.mean(self.recall)*100
    
    def precision_binary(self, pos_class_index=0):
        tp = self.cm[pos_class_index, pos_class_index]
        fp = np.sum(self.cm[:, pos_class_index]) - tp 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        return precision * 100

    def recall_binary(self, pos_class_index=0):
        tp = self.cm[pos_class_index, pos_class_index]
        fn = np.sum(self.cm[pos_class_index, :]) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return recall * 100

    def plotConfusionMatrix(self,title="Confusion Matrix"):
        plt.figure(figsize=(10, 7))
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        plt.show()

    def plotPrecisionRecall(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.labels, self.precision, label='Precision')
        plt.plot(self.labels, self.recall, label='Recall')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Precision and Recall')
        plt.legend()
        plt.show()
    
    def plotAccuracy(self,accuracy):
        plt.figure(figsize=(10, 7))
        plt.bar(['Accuracy'], [accuracy])
        plt.ylabel('Score')
        plt.title('Accuracy')
        plt.show()