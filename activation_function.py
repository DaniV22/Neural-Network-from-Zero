from loss import LossCategoricalCrossEntropy
import numpy as np

class ActivationReLU:

    '''
    A class to implement a Rectified Linear Unit activation function (ReLU)

    ATTRIBUTES :

        name :              the name of the activation function
        inputs :            the input received by the function
        output :            the output of the function
        gradient_inputs :   gradient with respect to inputs

    METHODS :

        forward :   performs a forward pass of the function
        backward :  performs a backward pass of the function
    '''

    def __init__(self):

        self.name = 'relu'

    def forward(self, inputs):

        '''
        Performs a forward pass of the activation function. It calculates the output
        taking the positive part of the argument (inputs)
        '''

        self.inputs = inputs
        self.output = np.maximum(0, inputs) #ReLU function

    def backward(self, previous_gradient):

        '''
        Performs a backward pass of the layer (backpropagation). It uses the
        values of the gradient of the previous process (layer, loss,...)
        to calculate the gradient with respect to the inputs
        '''

        self.gradient_inputs = previous_gradient.copy()
        self.gradient_inputs[self.inputs <= 0] = 0

class ActivationSoftmax:

    '''
    A class to implement a normalized exponential activation function (softmax)

    ATTRIBUTES :

        name :              the name of the activation function
        inputs :            the input received by the function
        output :            the output of the function
        gradient_inputs :   gradient with respect to inputs

    METHODS :

        forward :   performs a forward pass of the function
        backward :  performs a backward pass of the function
    '''
    def __init__(self):
        self.name = 'softmax'

    def forward(self, inputs):

        '''
        Performs a forward pass of the activation function. It calculates the output
        taking the positive part of the argument (inputs)
        '''

        self.inputs = inputs

        exponential_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalized_values = exponential_values / np.sum(exponential_values, axis=1,keepdims=True)
        self.output = normalized_values

    def backward(self, previous_gradient):

        '''
        Performs a backward pass of the layer (backpropagation). It uses the
        values of the gradient of the previous process (layer, loss,...)
        to calculate the gradient with respect to the inputs
        '''

        self.gradient_inputs = np.empty_like(previous_gradient)

        for index, (single_output, single_gradient_values) in enumerate(zip(self.output, previous_gradient)):

            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.gradient_inputs[index] = np.dot(jacobian_matrix, single_gradient_values)


class ActivationSoftmaxLossCategoricalCrossentropy():

    '''
    A class that combines the Softmax activation function with the Categorical
    Cross Entropy loss function. It is very useful to use in the last layer since
    the backward pass can be done in a single step in a much more efficient way.

    ATTRIBUTES:
        
        name :          the name of the activation function
        activation :    an activation function object (Softmax)
        loss :          a loss function object (Categorical Cross Entropy)
        output :        the output of the activation + loss functions

    '''

    def __init__(self):
        self.name = 'softmax_cross_entropy'
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs):

        '''
        Performs a forward pass of (only) the activation function.
        '''
        #Activation function
        self.activation.forward(inputs)

        self.output = self.activation.output

    def backward(self, previous_gradient, y_true):

        '''
        Performs a backward pass of the activation and the loss function (backpropagation),
        in a single step. The resulting gradients are much simpler than the ones resulting
        from the activation and loss separately
        '''

        # Number of samples
        samples = len(previous_gradient)

        #If labels are one-hot encoded
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.gradient_inputs = previous_gradient.copy()

        #Gradient
        self.gradient_inputs[range(samples), y_true] -= 1

        #Normalize gradient
        self.gradient_inputs = self.gradient_inputs / samples

class ActivationLinear:

    '''
    A class to implement a Linear Activation function for regression

    ATTRIBUTES

        name :              the name of the activation function
        inputs :            the input received by the function
        output :            the output of the function
        gradient_inputs :   gradient with respect to inputs

    METHODS :

        forward :   performs a forward pass of the function
        backward :  performs a backward pass of the function
    '''

    def __init__(self):
        self.name = 'linear'

    def forward(self, inputs):

        '''
        Performs a forward pass of the activation function. Since it is
        linear, the outputs are equal to the inputs
        '''

        self.inputs = inputs
        self.output = inputs

    def backward(self, previous_gradient):

        '''
        Performs a backward pass of the activation function. Since the
        derivative is 1, the gradient is the previous gradient
        '''
        self.gradient_inputs = previous_gradient.copy()
