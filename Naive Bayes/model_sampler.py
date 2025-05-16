from naive_bayes import NaiveBayes
import numpy as np
from calculate_accuracy import CalculateAccuracy
import matplotlib.pyplot as plt
from computation import randomColorGenerator

class NaiveBayesSampler:
    def __init__(self, labels, title, laplaceRange = [0], logProb = False, plotCM = False, classificationType='binary'):
        self.laplaceRange = laplaceRange
        self.logProb = logProb
        self.labels = labels
        self.title = title
        self.accuracies = []
        self.precision = []
        self.recall = []
        self.plotCM = plotCM
        self.classificationType = classificationType
    
    def createModel(self, trainData, bow, laplaceFactor):
        model = NaiveBayes(laplaceFactor=laplaceFactor, logProb=self.logProb)
        model.fit(trainData=trainData, bow=bow)
        return model

    def run(self,model,X_test):
        return model.predict(X_test)
    
    def sampler(self, trainData, X_test, y_test, bow):
        '''
        Used to run the model with different laplace factors
        '''
        for laplaceFactor in self.laplaceRange:
            model = NaiveBayes(laplaceFactor=laplaceFactor,logProb=True)
            model.fit(trainData=trainData,bow=bow)
            pred = model.predict(X_test)
            accObj = CalculateAccuracy(test=y_test,pred=pred,labels=self.labels)

            print("lf:",laplaceFactor, " Accuracy: ",accObj.accuracy())
            accObj.confusion_matrix()
            if self.plotCM:
                accObj.plotConfusionMatrix(title="Confusion Matrix for Laplace Factor: "+str(laplaceFactor))
            self.accuracies.append(accObj.accuracy())
            self.precision.append(accObj.precision_binary() if self.classificationType == 'binary' else accObj.precision())
            self.recall.append(accObj.recall_binary() if self.classificationType == 'binary' else accObj.recall())

    def superimposePrint(self):
        self.plotMetricVsLaplaceFactor(metricValues=[self.accuracies,self.precision,self.recall],
                                        metricNames=["Accuracy","Precision","Recall"],
                                        plotTile="All metric vs Laplace Factor",annotate=False)

    def plotAccuracy(self):
        self.plotMetricVsLaplaceFactor(metricValues=[self.accuracies],
                                        metricNames=["Accuracy"],
                                        plotTile="Accuracy vs Laplace Factor")

    def plotPrecision(self):
        self.plotMetricVsLaplaceFactor(metricValues=[self.precision],
                                        metricNames=["Precision"],
                                        plotTile="Precision vs Laplace Factor")
    
    def plotRecall(self):
        self.plotMetricVsLaplaceFactor(metricValues=[self.recall],
                                        metricNames=["Recall"],
                                        plotTile="Recall vs Laplace Factor")

    def plotMetricVsLaplaceFactor(self,metricValues=[],metricNames=[],yLabel="",plotTile="",annotate=True):
        plt.figure(figsize=(10, 7))
        # plot all values in one graph
        for i in range(len(metricValues)):
            x_positions = range(len(self.laplaceRange))
            color = randomColorGenerator()
            plt.plot(x_positions, metricValues[i], marker='o',color=color,label=metricNames[i])
            plt.xticks(x_positions, labels=[str(val) for val in self.laplaceRange])
            plt.grid()

            if annotate:
                for x, y in zip(x_positions, metricValues[i]):
                    plt.text(
                        x, y, f"{y:.2f}",
                        fontsize=10, ha='center', va='bottom',
                        bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2')
                )

        plt.xlabel(r'α (Laplace Factor/ Alpha Value)', fontsize=12, fontweight='light')
        plt.ylabel('Metric '+yLabel, fontsize=12, fontweight='light')
        plt.title(plotTile+': '+self.title)
        plt.legend()
        plt.show()

        