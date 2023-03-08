import numpy as np

class Loss:

    '''
    A Class to calculate the data and regularization losses
    given model output and ground truth values

    METHODS:

        calculate :     calculates the data loss
    
    '''
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

class LossCategoricalCrossEntropy(Loss):

    '''
    A class to calculate the Loss of the network according to Categorical
    Cross Entropy Loss function and to perform the backward pass.

    ATTRIBUTES:

        gradient_inputs : gradient with respect to inputs

    METHODS:

        forward :   performs a forward pass of the loss function
        backward :  performs a backward pass of the loss function
    '''

    def forward(self, y_pred, y_true):

        '''
        Performs a forward pass according to the Categorical Cross Entropy
        Loss function. The target values can be given either in a one-hot
        encoding format or with the actual categorical labels.
        '''

        #Number of samples
        samples = len(y_pred)

        #Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        #Categorical labels
        if len(y_true.shape) == 1:
            confidences = y_pred_clipped[range(samples), y_true]

        #One-hot encoded labels
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(confidences)

    def backward(self, previous_gradient, y_true):

        '''
        Performs a backward pass according to the Categorical Cross Entropy
        Loss function. 
        '''

        #Number of samples
        samples = len(previous_gradient)

        #Number of labels in every sample
        labels = len(previous_gradient[0])

        #Turn labels into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        #Calculate gradient and normalize it
        self.gradient_inputs = -y_true / (previous_gradient * samples)

class LossMeanSquaredError(Loss):

    def forward(self, y_pred, y_true):

        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        return sample_losses
    
    def backward(self, previous_gradient, y_true):

        samples = len(previous_gradient)

        num_outputs = len(previous_gradient[0])

        #Gradient
        self.gradient_inputs = -2 * (y_true - previous_gradient) / num_outputs

        #Normalization
        self.gradient_inputs = self.gradient_inputs / samples        



        