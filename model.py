import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class Model:

    '''
    A Class to implement a simple Neural Network model

    ATTRIBUTES :

        layers :    an array that includes the layer and activation function objects
        loss :      the loss function object used in the model       
        optimizer : the optimizer object used in the model
        accuracy :  the accuracy object used in the model

    METHODS :

        add_layer :         adds a layer (layer or activation function) to the layers attribute
        set :               sets the loss, optimizer and accuracy of the neural network
        forward :           performs a complete forward pass of the layers in the neural network
        backward :          performs a complete bacward pass (including loss) of the neural network
        train :             trains the model and updates parameters of the neural network
        validate :          validates the model with test data
        predict :           predicts the output of the network for a given input
        training_animation: animates the training process of the network (only for regression) 
    '''

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        '''
        Adds a layer to the layers attribute. We include in layers the activation function.
        The layer of neurons and the activation function should be alternated.

        '''
        self.layers.append(layer)

    def set(self, loss_function, optimizer, accuracy):
        '''
        Sets the loss function, optimizer and the accuracy object of the model. The loss function
        and the accuracy must be in agreement with each other (both for regression or classification).
        '''
        self.loss = loss_function
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    def forward(self, X):

        '''
        Performs a complete forward pass of the Neural Network given an input X.
        Returns the output (or prediction) of the model
        '''

        #"Output" of the Input Layer
        previous_output = X

        for layer in self.layers:
            layer.forward(previous_output)
            previous_output = layer.output  #The output of the layer is the input of the next one

        output = layer.output

        return output
    
    def backward(self, output, y):

        '''
        Performs a complete backward pass of the Neural network (including the loss function)
        '''

        #If last activation function is softmax cross entropy, we can
        #optimize the backward pass in a single step
        if self.layers[-1].name == 'softmax_cross_entropy':
            self.layers[-1].backward(output, y)
            previous_gradient = self.layers[-1].gradient_inputs

        else:
            self.loss.backward(output, y)
            self.layers[-1].backward(self.loss.gradient_inputs)
            previous_gradient = self.layers[-1].gradient_inputs

        #Backward pass of the remaining layers
        for layer in reversed(self.layers[:-1]):
            layer.backward(previous_gradient)
            previous_gradient = layer.gradient_inputs


    def train(self, X, y, epochs, return_predictions = False):

        '''
        Trains the model with the training data X and y, for the desired number of epochs.
        Return_predictions is used to animate the training process during the epochs
        
        '''

        for epoch in range(epochs):

            #Output (predictions) of the Network
            output = self.forward(X)

            if self.accuracy.type == 'classification':
                prediction = np.argmax(output, axis=1)
            
            else:
                prediction = output

            #Loss of the model
            loss = self.loss.forward(output, y)

            #Accuracy from output 
            accuracy = self.accuracy.get_accuracy(prediction, y)

            #Printing training process results every 100 epochs (if not return predictions)
            if not return_predictions:
                if not epoch % 100:
                    print(f'epoch: {epoch}, ' +
                        f'acc: {round(accuracy, 3)}, ' + 
                        f'loss : {round(loss[0], 3)}')
      
            #Backward pass     
            self.backward(output, y)

            #Updating learning rate of the optimizer
            self.optimizer.update_learning_rate()

            #Updating weights and biases of the layers (of neurons)
            for layer in self.layers[::2]:
                self.optimizer.update_parameters(layer)
            
            #Increase iteration by one unit
            self.optimizer.iteration()

            #Returning predictions
            if return_predictions:
                return X, y, output
            
    def validate(self, X_test, y_test):

        '''
        Validates the model with X and y data, after the training process
        and prints the accuracy and loss.
        '''

        #Output (prediction of the model)
        output = self.forward(X_test)

        if self.accuracy.type == 'classification':
            prediction = np.argmax(output, axis=1)
            
        else:
            prediction = output

        #Loss of the model
        loss = self.loss.calculate(output, y_test)

        #Accuracy of the validation process
        accuracy = self.accuracy.get_accuracy(prediction, y_test)

        print('Validation ' +
                        f'acc: {round(accuracy, 3)}, ' + 
                        f'loss : {round(loss, 3)}')
        
    def predict(self, X):
        '''
        Returns the prediction of a given input X
        '''

        output = self.forward(X)

        if self.accuracy.type == 'classification':
                return np.argmax(output, axis=1)
        
        return output
        
    def training_animation(self, X, y, max_epochs, anim_speed):

        '''
        Animates the training process of the Neural Network and produces a GIF.
        It should be used only with a REGRESSION task.
        '''

        figure, ax = plt.subplots()
        
        #Setting limits for x and y axis
        #change limits depending on the data
        ax.set_xlim(-1, 3)
        ax.set_ylim(-0.1, 1.1)
        
        line1,  = ax.plot(0, 0, label = 'True data') 
        line2, = ax.plot([], [], label = 'Predicted data')
        
        #Animation function
        def animation_function(i):

            #Single step of the train process
            _, _, y_pred_i = self.train(X, y, 1, return_predictions=True)

            #Real data
            line1.set_xdata(X)
            line1.set_ydata(y)

            #Predicted data
            line2.set_xdata(X)
            line2.set_ydata(y_pred_i)

            plt.title(f'Number of epochs: {i}')
            plt.legend()

            return line1, line2
        
        #Animating
        anim = animation.FuncAnimation(figure,
                                func = animation_function,
                                frames = np.array(range(max_epochs)),
                                interval = anim_speed)
        
        anim.save('train_animation.gif')
