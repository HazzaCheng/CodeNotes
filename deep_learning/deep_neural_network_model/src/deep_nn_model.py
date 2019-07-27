#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-07-04

"""
A simple deep neural network with L layers, use relu or sigmoid activation functions.
"""

import numpy as np

from dnn_utils import sigmoid, relu, relu_backward, sigmoid_backward, dictionary_to_vector, gradients_to_vector, \
    vector_to_dictionary
import matplotlib.pyplot as plt

from initialize_parameters import initialize_parameters_zeros, initialize_parameters_random, initialize_parameters_he
from testCases import linear_forward_test_case, linear_activation_forward_test_case, L_model_forward_test_case_2hidden, \
    compute_cost_test_case, linear_backward_test_case, linear_activation_backward_test_case, L_model_backward_test_case, \
    print_grads, update_parameters_test_case
from update_parameters import update_parameters_with_gd, update_parameters_with_momentum, update_parameters_with_adam, \
    initialize_velocity, initialize_adam


class DeepNeuralNetworkModel:

    def __init__(self,
                 layers_dims,
                 parameters_initialization_method="random",
                 parameters_update_optimizer="gd",
                 lambd=0,
                 beta=0.9,
                 beta1=0.9,
                 beta2=0.999,
                 keep_prob=1.0,
                 learning_rate=0.0075,
                 num_iterations=3000,
                 print_cost=False,
                 gradient_check=False,
                 random_state=42
                 ):
        self.layers_dims = layers_dims
        self.parameters_initialization_method = parameters_initialization_method
        self.parameters_update_optimizer = parameters_update_optimizer
        self.keep_prob = keep_prob
        self.lambd = lambd
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.gradient_check = gradient_check
        self.random_state = random_state

        self.parameters = {}
        self.X = None
        self.Y = None

    def __initialize_parameters(self):
        if self.parameters_initialization_method == "zeros":
            self.parameters = initialize_parameters_zeros(self.layers_dims)
        elif self.parameters_initialization_method == "random":
            self.parameters = initialize_parameters_random(self.layers_dims, self.random_state)
        elif self.parameters_initialization_method == "he":
            self.parameters = initialize_parameters_he(self.layers_dims, self.random_state)
        else:
            raise Exception("No such parameters initialization methond.")

        L = len(self.layers_dims)  # number of layers in the network

        for l in range(1, L):
            assert (self.parameters['W' + str(l)].shape == (self.layers_dims[l], self.layers_dims[l - 1]))
            assert (self.parameters['b' + str(l)].shape == (self.layers_dims[l], 1))

    def __intialize_optimizer(self):
        # Initialize the parameter update optimizer
        if self.parameters_update_optimizer == "gd":
            pass  # no initialization required for gradient descent
        elif self.parameters_update_optimizer == "momentum":
            self.v = initialize_velocity(self.parameters)
        elif self.parameters_update_optimizer == "adam":
            self.t = 0
            self.v, self.s = initialize_adam(self.parameters)
        else:
            raise Exception("No such parameter update optimizer: {}".format(self.parameters_update_optimizer))

    def __linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input datasets): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = np.dot(W, A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def __linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input datasets): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.__linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.__linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
        else:
            raise Exception("No such activation method: {}".format(activation))

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def __forward_propagation(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        """

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, cache = self.__linear_activation_forward(A_prev,
                                                        parameters['W' + str(l)],
                                                        parameters['b' + str(l)],
                                                        activation="relu")
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.__linear_activation_forward(A,
                                                     parameters['W' + str(L)],
                                                     parameters['b' + str(L)],
                                                     activation="sigmoid")
        caches.append(cache)

        assert (AL.shape == (1, X.shape[1]))

        return AL, caches

    def __forward_propagation_with_dropout(self, X, parameters):
        """
        Implements the forward propagation: [LINEAR -> RELU + DROPOUT]*(L-1) -> LINEAR -> SIGMOID.
        """
        np.random.seed(self.random_state)

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, cache = self.__linear_activation_forward(A_prev,
                                                        parameters['W' + str(l)],
                                                        parameters['b' + str(l)],
                                                        activation="relu")
            D = np.random.rand(A.shape[0], A.shape[1])
            D = D < self.keep_prob
            A = np.multiply(D, A)
            A = A / self.keep_prob
            cache = (cache[0], cache[1], D)
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.__linear_activation_forward(A,
                                                     parameters['W' + str(L)],
                                                     parameters['b' + str(L)],
                                                     activation="sigmoid")
        caches.append(cache)

        assert (AL.shape == (1, X.shape[1]))

        return AL, caches

    def __compute_cost(self, AL):
        """
        Implement the cost function, support regularization.
        """

        m = self.Y.shape[1]

        # Compute loss from aL and y.
        cross_entropy_cost = -1 / m * np.sum(self.Y * np.log(AL) + (1 - self.Y) * (np.log(1 - AL)))

        # To make sure your cost's shape is what we expect (e.g. this turns
        # [[17]] into 17).
        cost = np.squeeze(cross_entropy_cost)
        assert (cost.shape == ())

        if self.lambd:
            L = len(self.layers_dims)
            W_square_sum = [np.sum(np.square(self.parameters["W" + str(i)])) for i in range(1, L)]
            L2_regularization_cost = 1 / m * self.lambd / 2 * (np.sum(W_square_sum))
            cost = cost + L2_regularization_cost

        return cost

    def __linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T) + self.lambd / m * W
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def __linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache[0], cache[1]

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
        else:
            raise Exception("No such activation method: {}.".format(activation))
        dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def __backward_propagation(self, AL, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = self.Y.reshape(AL.shape)

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL,
        # current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] \
            = self.__linear_activation_backward(dAL, current_cache, 'sigmoid')

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.__linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, 'relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def __backward_propagation_with_dropout(self, AL, caches):
        """
        Implements the backward propagation of our baseline model to which we added dropout.
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = self.Y.reshape(AL.shape)

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL,
        # current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] \
            = self.__linear_activation_backward(dAL, current_cache, 'sigmoid')

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            D = current_cache[2]
            dA_prev_temp, dW_temp, db_temp = self.__linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, 'relu')
            dA_prev_temp = np.multiply(dA_prev_temp, D)
            dA_prev_temp = dA_prev_temp / self.keep_prob
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def __update_parameters(self, grads):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """

        # number of layers in the neural network
        L = len(self.parameters) // 2

        if self.parameters_update_optimizer == "gd":
            parameters = update_parameters_with_gd(self.parameters, grads, self.learning_rate)
        elif self.parameters_update_optimizer== "momentum":
            parameters, v = update_parameters_with_momentum(self.parameters, grads, self.v, self.beta, self.learning_rate)
        elif self.parameters_update_optimizer == "adam":
            self.t = self.t + 1  # Adam counter
            parameters, v, s = update_parameters_with_adam(self.parameters, grads, self.v, self.s,
                                                           t, self.learning_rate, self.beta1, self.beta2)
        else:
            raise Exception("No such parameter update optimizer: {}".format(self.parameters_update_optimizer))

    def predict(self, X, y):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        # number of layers in the neural network
        n = len(self.parameters) // 2
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = self.__forward_propagation(X)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        # print results
        # print ("predictions: " + str(p))
        # print ("true labels: " + str(y))
        print("Accuracy: " + str(np.sum((p == y) / m)))

        return p

    def fit(self, X, Y):
        assert (self.lambd == 0 or self.keep_prob == 1)  # it is possible to use both L2 regularization and dropout,
        assert (0 < self.keep_prob <= 1)
        self.X = X
        self.Y = Y
        np.random.seed(self.random_state)
        costs = []  # keep track of cost

        # Parameters initialization. (≈ 1 line of code)
        self.__initialize_parameters()

        # Loop (gradient descent)
        for i in range(0, self.num_iterations):

            # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
            if self.keep_prob == 1:
                AL, cache = self.__forward_propagation(self.X, self.parameters)
            else:
                AL, cache = self.__forward_propagation_with_dropout(self.X, self.parameters)

            # Cost function
            cost = self.__compute_cost(AL)

            # Backward propagation.
            # but this assignment will only explore one at a time
            if self.keep_prob == 1:
                grads = self.__backward_propagation(AL, cache)
            else:
                grads = self.__backward_propagation_with_dropout(AL, cache)

            # Update parameters.
            self.__update_parameters(grads)

            # gradients check
            if self.keep_prob == 1 and self.gradient_check:
                if i % 100 == 0:
                    diff = self.__check_gradient(grads)

            # Print the cost every 100 training iterations
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if self.print_cost and i % 100 == 0:
                costs.append(cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()

        return self

    def __check_gradient(self, gradients, epsilon=1e-7):
        """
        Checks gradient.
        """

        parameters_values, _ = dictionary_to_vector(self.parameters)
        grad = gradients_to_vector(gradients)
        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))

        # Compute gradapprox
        for i in range(num_parameters):
            thetaplus = np.copy(parameters_values)
            thetaplus[i][0] = thetaplus[i][0] + epsilon
            AL1, _ = self.__forward_propagation(self.X, vector_to_dictionary(thetaplus, self.layers_dims))
            AL3, _ = self.__forward_propagation(self.X, self.parameters)
            J_plus[i] = self.__compute_cost(AL1)
            thetaminus = np.copy(parameters_values)
            thetaminus[i][0] = thetaminus[i][0] - epsilon
            AL2, _ = self.__forward_propagation(self.X, vector_to_dictionary(thetaminus, self.layers_dims))
            AL4, _ = self.__forward_propagation(self.X, self.parameters)
            J_minus[i] = self.__compute_cost(AL2)

            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
            if i % 1000 == 0:
                print(i)

        numerator = np.linalg.norm(gradapprox - grad)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
        difference = numerator / denominator

        if difference > 2e-7:
            print("There is a mistake in the backward propagation! difference = " + str(difference))
        else:
            print("Your backward propagation works perfectly fine! difference = " + str(difference))

        return difference