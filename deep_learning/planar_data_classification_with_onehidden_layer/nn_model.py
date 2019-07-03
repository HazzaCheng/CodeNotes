#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-07-03
# A simple shallow neural network model for planar data classification.


from planar_utils import sigmoid
from testCases import *


class SimpleNeuralNetWorkModel:

    def __layer_sizes(self, X, Y):
        """
        Arguments:
        X -- input dataset of shape (input size, number of examples)
        Y -- labels of shape (output size, number of examples)

        Returns:
        n_x -- the size of the input layer
        n_y -- the size of the output layer
        """
        # size of input layer
        n_x = X.shape[0]
        # size of output layer
        n_y = Y.shape[0]

        return (n_x, n_y)

    def __initialize_parameters(slef, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """

        # we set up a seed so that your output matches ours although the initialization is random.
        np.random.seed(2)

        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def __forward_propagation(slef, X, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        # Implement Forward Propagation to calculate A2 (probabilities)
        # Z1 = np.dot(W1, X) + b1
        # A1 = np.tanh(Z1)
        # Z2 = np.dot(W2, A1) + b2
        # A2 = sigmoid(Z2)
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        assert (A2.shape == (1, X.shape[1]))

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return A2, cache

    def __compute_cost(self, A2, Y):
        """
        Computes the cross-entropy cost given in equation (13)

        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost given equation (13)
        """

        m = Y.shape[1]  # number of example
        # Compute the cross-entropy cost
        # print(A2)
        logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1-Y), np.log((1-A2)))
        cost = -1/m * np.sum(logprobs)

        # makes sure cost is the dimension we expect.
        cost = float(np.squeeze(cost))
        # E.g., turns [[17]] into 17
        assert (isinstance(cost, float))

        return cost

    def __backward_propagation(self, parameters, cache, X, Y):
        """
        Implement the backward propagation using the instructions above.

        Arguments:
        parameters -- python dictionary containing our parameters
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]

        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = parameters['W1']
        W2 = parameters['W2']

        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache['A1']
        A2 = cache['A2']

        # Backward propagation: calculate dW1, db1, dW2, db2.
        dZ2 = A2 - Y
        dW2 = 1 / m * np.dot(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = 1 / m * np.dot(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads

    def __update_parameters(self, parameters, grads, learning_rate=1.2):
        """
        Updates parameters using the gradient descent update rule given above

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients

        Returns:
        parameters -- python dictionary containing your updated parameters
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        # Retrieve each gradient from the dictionary "grads"
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        # Update rule for each parameter
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def predict(self, parameters, X):
        """
        Using the learned parameters, predicts a class for each example in X

        Arguments:
        parameters -- python dictionary containing your parameters
        X -- input data of size (n_x, m)

        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """

        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        A2, cache = self.__forward_propagation(X, parameters)
        predictions = np.array([1 if x > 0.5 else 0 for x in A2.reshape(-1, 1)]).reshape(A2.shape)

        return predictions

    def getModel(self, X, Y, n_h, num_iterations=10000, learning_rate=1.2, print_cost=False):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(3)
        # Initialize parameters, then retrieve W1, b1, W2, b2.
        # Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
        n_x, n_y = self.__layer_sizes(X, Y)
        parameters = self.__initialize_parameters(n_x, n_h, n_y)
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = self.__forward_propagation(X, parameters)

            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = self.__compute_cost(A2, Y)
            # print(cost)

            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = self.__backward_propagation(parameters, cache, X, Y)

            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = self.__update_parameters(parameters, grads, learning_rate)

            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters

