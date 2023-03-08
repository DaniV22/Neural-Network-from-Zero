import numpy as np

class Accuracy:

    '''
    A class to calculate the accuracy of the model depending on if it is
    a classification problem or a regression one.

    ATTRIBUTES :

        type :          a string with type of the problem ('classification' or 'regression')
        fraction_std :  a number to control the precision used when calculating the accuracy (only when it is a regression problem)

    METHODS :

        get_accuracy :  calculates the accuracy of the model, based on the predictions and y true
        compare :       compare the arrays of predictions and y true to get whether they are equal or not
    '''

    def __init__(self, type, fraction_std = 200):
        self.type = type
        self.fraction_std = 1 / fraction_std

    def get_accuracy(self, y_pred, y):

        '''
        A method that calculates the accuracy of the model. It uses the compare method
        to get an array of booleans and calculates the accuracy as the mean of true values
        '''

        comparisions = self.compare(y_pred, y)
        accuracy = np.mean(comparisions)

        return accuracy
    
    def compare(self, y_pred, y):

        '''
        A method that returns an array of booleans. A value is true if the prediction and
        y are equal (categorical case) or the prediction is inside a certain range of values (regression case),
        defined by the precision of the accuracy.
        '''
        
        if self.type == 'regression':
            return np.absolute(y_pred - y) < np.std(y)*self.fraction_std

        elif self.type == 'classification':
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            return y_pred == y
        
        else:
            print('Invalid type')
            return 0

