import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as np
import itertools
import time


# TODO: CIRCUIT TRAINING GOES HERE


def train_circuit(circuit, parameter_shape, X_train, Y_train, rate_type='accuracy', **kwargs):
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
        (p,i,e,w): The number of parameters, the inference time (time to evaluate the accuracy), error rate (accuracy on the test set)
    """

    # Use this array to make a prediction for the labels of the data in X_test
    def hinge_loss(labels, predictions, type='L2'):
        loss = 0
        for l, p in zip(labels, predictions):
            if type == 'L1':
                loss = loss + np.abs(l - p)  # L1 loss
            elif type == 'L2':
                loss = loss + (l - p) ** 2  # L2 loss
        loss = loss / len(labels)
        return loss

    def accuracy(labels, predictions):

        loss = 0
        tol = 0.05
        # tol = 0.1
        for l, p in zip(labels, predictions):
            if abs(l - p) < tol:
                loss = loss + 1
        loss = loss / len(labels)

        return loss

    def cost_fcn(params, circuit=None, ang_array=[], actual=[]):
        '''
        use MAE to start
        '''
        # labels = {2: -1, 1: 1, 0: 0}
        # n = len(ang_array[0])
        # w = params[-n:]
        # theta = params[:-n]

        predictions = [circuit(params, x) for x in ang_array]
        return hinge_loss(actual, predictions)


    var = np.random.random(parameter_shape) - 2.5
    batch_size = kwargs['batch_size']
    num_train = len(Y_train)
    validation_size = 3 * kwargs['batch_size']
    opt = qml.AdamOptimizer(kwargs['learning_rate'])
    start = time.time()
    for _ in range(kwargs['nsteps']):
        batch_index = np.random.randint(0, num_train, (batch_size,))
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        var, cost = opt.step_and_cost(lambda v: cost_fcn(v, circuit, X_train_batch, Y_train_batch), var)
        print(cost)
    end = time.time()
    cost_time = (end - start)

    w = var[-X_train.shape[1]:]
    theta = var[:-X_train.shape[1]]

    if rate_type == 'accuracy':
        validation_batch = np.random.randint(0, num_train, (validation_size,))
        X_validation_batch = X_train[validation_batch]
        Y_validation_batch = Y_train[validation_batch]
        start = time.time()  # add in timeit function from Wbranch
        predictions = [circuit(theta, x) for x in X_validation_batch]
        end = time.time()
        inftime = (end - start) / len(X_validation_batch)
        err_rate = 1.0 - accuracy(predictions, Y_validation_batch)
    elif rate_type == 'batch_cost':
        err_rate = cost
        inftime = cost_time
    # QHACK #
    W_ = np.abs((100. - len(var)) / (100.)) * np.abs((100. - inftime) / (100.)) * (1. / err_rate)
    return len(var), inftime, err_rate, W_, var

#
# def loop_over_hyperparameters(circuit, n_params, X_train, Y_train, batch_sets, learning_rates, **kwargs):
#     """
#     together with the function train_circuit(...) this executes lines 7-8 in the Algorithm 1 pseudo code of (de Wynter 2020)
#     """
#     hyperparameter_space = list(itertools.product(batch_sets, learning_rates))
#     Wmax = 0.0
#     s = kwargs.get('nsteps', None)
#     rate_type = kwargs.get('rate_type', None)
#
#     for idx, sdx in hyperparameters:
#         p, i, er, wtemp, weights = train_circuit(circuit, n_params, X_train, Y_train, X_test, Y_test, s=s,
#                                                  batch_size=idx, rate_type=rate_type, learning_rate=sdx)
#         if wtemp >= Wmax:
#             Wmax = wtemp
#             saved_weights = weights
#     return Wmax, saved_weights
