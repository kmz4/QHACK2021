#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np
import sklearn as skl
import autograd.numpy as np
import itertools
import time

def train_circuit(circuit,n_params,X_train,Y_train,X_test,Y_test,**kwargs):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        circuit (qml.QNode): A circuit that you want to train
        X_train (np.ndarray): An array of floats of size (M, n) to be used as training data.
        Y_train (np.ndarray): An array of size (M,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (B, n) to serve as testing data.
        kwargs: hyperparameters for the training (steps, batch_size, learning_rate)

    Returns:
        (p,i,e): The number of parameters, the inference time (time to evaluate the accuracy), error rate (accuracy on the test set)
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #

    from autograd.numpy import exp,tanh

    def hinge_loss(labels, predictions,type='L2'):
        loss = 0
        for l, p in zip(labels, predictions):
            if type=='L1':
                loss = loss + np.abs(l - p) # L1 loss
            elif type=='L2':
                loss = loss + (l - p) ** 2 # L2 loss
        loss = loss/len(labels)
        return loss

    def accuracy(labels, predictions):

        loss = 0
        tol = 0.05
        #tol = 0.1
        for l, p in zip(labels, predictions):
            if abs(l - p) < tol:
                loss = loss + 1
        loss = loss / len(labels)

        return loss

    def cost_fcn(params,circuit=None,ang_array=[], actual=[]):
        '''
        use MAE to start
        '''
        labels = {2:-1,1:1,0:0}
        n = len(ang_array[0])
        w = params[-n:]
        theta = params[:-n]
        predictions = [2.*(1.0/(1.0+exp(np.dot(-w,circuit(theta, angles=x)))))- 1. for x in ang_array]
        return hinge_loss(actual, predictions)

    var = np.hstack((np.zeros(n_params),5*np.random.random(X_train.shape[1])-2.5))
    steps = kwargs['s']
    batch_size = kwargs['batch_size']
    num_train = len(Y_train)
    validation_size = int(num_train//2)
    opt = qml.AdamOptimizer(kwargs['learning_rate'])

    for _ in range(steps):
        batch_index = np.random.randint(0, num_train, (batch_size,))
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]

        var,cost = opt.step_and_cost(lambda v: cost_fcn(v, circuit,X_train_batch, Y_train_batch), var)
    w = var[-X_train.shape[1]:]
    theta = var[:-X_train.shape[1]]
    start = time.time() # add in timeit function from Wbranch
    predictions=[int(np.round(2.*(1.0/(1.0+exp(np.dot(-w,circuit(theta, angles=x)))))- 1.,0)) for x in X_test]
    end = time.time()
    inftime = end-start
    err_rate = accuracy(predictions,Y_test)
    # QHACK #
    W_ = len(var)*inftime*(1./err_rate)
    return len(var),inftime,err_rate,W_


def classify_data(X_train,Y_train,X_test,Y_test,**kwargs):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #

    from autograd.numpy import exp,tanh

    def statepreparation(a):
        qml.templates.embeddings.AngleEmbedding(a, wires=range(3), rotation='Y')

    def layer(W):
        qml.templates.layers.BasicEntanglerLayers(W, wires=range(3), rotation=qml.ops.RY)

    def hinge_loss(labels, predictions,type='L2'):
        loss = 0
        for l, p in zip(labels, predictions):
            if type=='L1':
                loss = loss + np.abs(l - p) # L1 loss
            elif type=='L2':
                loss = loss + (l - p) ** 2 # L2 loss
        loss = loss/len(labels)
        return loss

    def accuracy(labels, predictions):

        loss = 0
        tol = 0.05
        #tol = 0.1
        for l, p in zip(labels, predictions):
            if abs(l - p) < tol:
                loss = loss + 1
        loss = loss / len(labels)

        return loss

    def cost_fcn(params,circuit=None,ang_array=[], actual=[]):
        '''
        use MAE to start
        '''
        labels = {2:-1,1:1,0:0}
        w = params[-3:]
        theta = params[:-3]
        predictions = [2.*(1.0/(1.0+exp(np.dot(-w,circuit(theta, angles=x)))))- 1. for x in ang_array]
        return hinge_loss(actual, predictions)

    dev = qml.device("default.qubit", wires=3)
    @qml.qnode(dev)
    def inside_circuit(params,angles=None):
        statepreparation(angles)
        W= np.reshape(params,(len(params)//3,3))
        layer(W)
        return qml.expval(qml.PauliZ(0)),qml.expval(qml.PauliZ(1)),qml.expval(qml.PauliZ(2))


    var = np.hstack((np.zeros(6),5*np.random.random(3)-2.5))
    steps = kwargs['s']
    batch_size = kwargs['batch_size']
    num_train = len(Y_train)
    validation_size = int(num_train//2)
    opt = qml.AdamOptimizer(kwargs['learning_rate'])

    for _ in range(steps):
        batch_index = np.random.randint(0, num_train, (batch_size,))
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]

        var,cost = opt.step_and_cost(lambda v: cost_fcn(v, inside_circuit,X_train_batch, Y_train_batch), var)

    # need timing values from computing predictions


    theta = var[:-3]
    w = var[-3:]
    start = time.time() # add in timeit function from Wbranch
    predictions=[int(np.round(2.*(1.0/(1.0+exp(np.dot(-w,inside_circuit(theta, angles=x)))))- 1.,0)) for x in X_test]
    end = time.time()
    inftime = end-start
    err_rate = accuracy(predictions,Y_test)
    # QHACK #
    W_ = len(var)*inftime*(1./err_rate)
    return len(var),inftime,err_rate,W_
