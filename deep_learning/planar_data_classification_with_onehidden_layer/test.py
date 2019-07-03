#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-07-03

import matplotlib.pyplot as plt
import numpy as np
import sklearn

from nn_model import SimpleNeuralNetWorkModel, nn_model_test_case
from planar_utils import load_planar_dataset, plot_decision_boundary
from testCases import predict_test_case

if __name__ == '__main__':
    """
    datasets.
    """
    X, Y = load_planar_dataset()
    # Visualize the data:
    plt.scatter(X[0, :], X[1, :], c=Y.flatten(), s=40, cmap=plt.cm.Spectral)
    plt.show()
    print('The shape of X is: ' + str(X.shape))
    print('The shape of Y is: ' + str(Y.shape))
    print('I have m = %d training examples!' % (X.shape[1]))

    """
    simple logistic regression
    """
    # Train the logistic regression classifier
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T.flatten())
    # Plot the decision boundary for logistic regression
    plt.title("Logistic Regression")
    plot_decision_boundary(lambda x: clf.predict(x), X, Y.flatten())
    plt.show()
    # Print accuracy
    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d ' % float(
        (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
        '% ' + "(percentage of correctly labelled datapoints)")

    """
    neural network model
    """
    network = SimpleNeuralNetWorkModel()

    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20, 50, 70, 100]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = network.getModel(X, Y, n_h, num_iterations=5000)
        plot_decision_boundary(lambda x: network.predict(parameters, x.T), X, Y.flatten())
        predictions = network.predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    plt.show()
    print()