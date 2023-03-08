import numpy as np

class Layer:

    '''
    A class to store the layer instances and to perform the possible
    actions of itself

    ATTRIBUTES :

        weights :           a matrix to store the weights of each neuron
        biases :            an array to store the biases of each neuron
        inputs :            the input received by the layer
        output :            the output of the layer
        gradient_weights :  gradient with respect to weights
        gradient_biases :   gradient with respect to biases
        gradient_inputs :   gradient with respect to inputs

    METHODS :

        forward :   performs a forward pass of the layer
        backward :  performs a backward pass of the layer
    '''

    def __init__(self, n_inputs, n_neurons):

        #Initializing weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):

        '''
        Performs a forward pass of the layer. It uses the given input,
        the weights and biases of the layer to calculate the output
        '''
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, previous_gradient):
        '''
        Performs a backward pass of the layer (backpropagation). It uses the
        values of the gradient of the previous process (activation, loss,...)
        to calculate the gradient with respect to weights, biases and inputs
        '''
        
        self.gradient_weights = np.dot(self.inputs.T, previous_gradient)
        self.gradient_biases = np.sum(previous_gradient, axis = 0, keepdims=True)
        self.gradient_inputs = np.dot(previous_gradient, self.weights.T)