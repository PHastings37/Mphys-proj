"""
This file includes the results class that will be imported into the main network code.

Rory Farwell and Patrick Hastings 22/03/2022
"""
import sklearn
from sklearn.metrics import confusion_matrix

class results :
    def __init__(self, expected, predicted) :
        self.expected = expected
        self.predicted = predicted

    def confusion_matrix(self):
        print(sklearn.metrics.confusion_matrix(self.expected, self.predicted))
        return sklearn.metrics.confusion_matrix(self.expected, self.predicted)
    
    def evaluate_results(self):
        self.true_positive_counter = 0
        self.true_negative_counter = 0
        self.false_positive_counter = 0
        self.false_negative_counter = 0
        for i in range(len(self.expected)) :
            if self.expected[i] == 1 and self.predicted[i] == 1 :
                self.true_positive_counter += 1
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> true positive')
            elif self.expected[i] == 0 and self.predicted[i] == 0 :
                self.true_negative_counter += 1
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> true negative')
            elif self.expected[i] == 0 and self.predicted[i] == 1 :
                self.false_positive_counter += 1 
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> false positive')
            elif self.expected[i] == 1 and self.predicted[i] == 0 :
                self.false_negative_counter += 1 
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> false negative')
        return self.true_positive_counter, self.true_negative_counter, self.false_positive_counter, self.false_negative_counter
