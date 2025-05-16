from computation import generateWordFreq

from collections import Counter
import math
from decimal import *

class TrainClassObject:
    def __init__(self,className,bow,prioriProb,laplaceFactor=0,logProb=False):
        self.className = className
        self.bow = bow
        self.freqMap = dict.fromkeys(bow,0)
        self.prioriProb = prioriProb
        self.laplaceFactor = laplaceFactor
        self.logProb = logProb
    
    def generateMultinomialFrequency(self,classData):
        for doc in classData:
            for word,count in generateWordFreq(doc):
                # if doc in self.freqMap.keys():
                # print(word,count,self.freqMap[word],"\n")
                self.freqMap[word] += count

    def computeTotalWords(self):
        self.totalCount = sum(self.freqMap.values())
    
    def calculatePosteriorProbality(self, testDoc):
        n_y = Decimal(self.totalCount + (self.laplaceFactor * len(self.bow)))
        
        if self.logProb:
            prob = Decimal(0)
            for word in testDoc:
                n_wi = Decimal(self.freqMap.get(word, 0) + self.laplaceFactor)
                prob += ((Decimal(n_wi).ln() - Decimal(n_y).ln()))
                # (n_wi/n_y).ln()
            return Decimal(self.prioriProb).ln() + prob
        else:
            prob = Decimal(1)
            for word in testDoc:
                n_wi = Decimal(self.freqMap.get(word, 0) + self.laplaceFactor)
                prob *= (n_wi/n_y)
            return self.prioriProb * prob
    
    def createFrequencyMatrix(self,classData):
        self.generateMultinomialFrequency(classData)
        self.computeTotalWords()

    



    
