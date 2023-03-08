import numpy as np

class OptimizerSGD:

    '''
    A class to implement the Stochastic Gradient Descent Optimizer (SGD)
    with a (decaying) learning rate and momentums

    ATTRIBUTES:

        learing_rate :  step size of each iteration (learing rate of the network)
        decay :         rate at which the learning rate decays during training
        iterations :    number of iterations that the optimizer has gone through
        momentum :      parameter to accelerate/decrease the optimization process (like inertia)

    METHODS:

        update_learning_rate :  updates the learning rate according to the decay of the Optimizer
        update_parameters :     updates weights and biases of the given layer
        iteration :             increases the iterations by one unit
    '''

    def __init__(self, learning_rate=1, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def update_learning_rate(self):
        '''
        Updates the learning rate according to the decay and the number of iterations. It should
        be called before the optimization process
        '''
        self.learning_rate = self.learning_rate * (1 / (1 + self.decay*self.iterations))

    def update_parameters(self, layer):

        '''
        Updates the biases and weights of the given layer, using the learning rate,
        decay and momentum of the Optimizer.
        '''

        #If layer does not contain momentum arrays, create them
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        #Weights updates
        weight_updates = self.momentum*layer.weight_momentums - self.learning_rate*layer.gradient_weights
        layer.weight_momentums = weight_updates

        #Bias updates
        bias_updates = self.momentum*layer.bias_momentums - self.learning_rate * layer.gradient_biases
        layer.bias_momentums = bias_updates

        # Update layer weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def iteration(self):
        self.iterations += 1

class OptimizerAdam:

    '''
    A class to implement the Adaptive Moment Estimation Optimizer (Adam)

    ATTRIBUTES:

        learing_rate :  step size of each iteration (learing rate of the network)
        decay :         rate at which the learning rate decays during training
        iterations :    number of iterations that the optimizer has gone through
        epsilon :       a small parameter to prevent dividing by zero
        beta_1 :        forgetting factor for gradient
        beta_2 :        forgetting factor of second moment of gradient

    METHODS:

        update_learning_rate :  updates the learning rate according to the decay of the Optimizer
        update_parameters :     updates weights and biases of the given layer
        iteration :             increases the iterations by one unit
    '''

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def update_learning_rate(self):
        self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay*self.iterations))

    def update_parameters(self, layer):

        #If layer does not contain second moments arrays
        if not hasattr(layer, 'weight_second_moments'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_second_moments = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_second_moments = np.zeros_like(layer.biases)

        #Update momentum  with current gradients
        layer.weight_momentums = self.beta_1*layer.weight_momentums + (1 - self.beta_1)*layer.gradient_weights
        layer.bias_momentums = self.beta_1*layer.bias_momentums + (1 - self.beta_1)*layer.gradient_biases

        #Corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1**(self.iterations + 1))

        #Update second moments with current squared gradients
        layer.weight_second_moments = self.beta_2*layer.weight_second_moments + (1 - self.beta_2)*layer.gradient_weights**2
        layer.bias_second_moments = self.beta_2*layer.bias_second_moments + (1 - self.beta_2)*layer.gradient_biases**2

        #Corrected second moments
        weight_second_moments_corrected = layer.weight_second_moments / (1 - self.beta_2**(self.iterations + 1))
        bias_second_moments_corrected = layer.bias_second_moments / (1 - self.beta_2**(self.iterations + 1))

        #Update parameters
        layer.weights += -self.learning_rate*weight_momentums_corrected / (np.sqrt(weight_second_moments_corrected) + self.epsilon)
        layer.biases += -self.learning_rate*bias_momentums_corrected / (np.sqrt(bias_second_moments_corrected) + self.epsilon)

    # Call once after any parameter updates
    def iteration(self):
        self.iterations += 1