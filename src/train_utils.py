import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as np
from autograd.numpy import exp
import itertools
import time


def hinge_loss(labels, predictions, type='L2'):
    loss = 0
    for l, p in zip(labels, predictions):
        if type == 'L1':
            loss = loss + np.abs(l - p)  # L1 loss
        elif type == 'L2':
            loss = loss + (l - p) ** 2  # L2 loss
    loss = loss / labels.shape[0]
    return loss


def ohe_accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += np.argmax(l) == np.argmax(p)
    return loss / labels.shape[0]

def wn_accuracy(labels, predictions):

    loss = 0
    #tol = 0.05
    tol = 0.1
    for l, p in zip(labels, predictions):
        if abs(l - p) < tol:
            loss = loss + 1
    loss = loss / labels.shape[0]

    return loss

def mse(labels, predictions):
    # print(labels.shape, predictions.shape)
    loss = 0
    for l, p in zip(labels, predictions):
        loss += np.sum((l - p) ** 2)
    return loss / labels.shape[0]


def train_circuit(circuit, parameter_shape,X_train, Y_train, batch_size, learning_rate,**kwargs):
    """
    train a circuit classifier
    Args:
        circuit (qml.QNode): A circuit that you want to train
        parameter_shape: A tuple describing the shape of the parameters. The first entry is the number of qubits,
        the second one is the number of layers in the circuit architecture. 
        X_train (np.ndarray): An array of floats of size (M, n) to be used as training data.
        Y_train (np.ndarray): An array of size (M,) which are the categorical labels
            associated to the training data.

        batch_size (int): Batch size for the circuit training.

        learning_rate (float): The learning rate/step size of the optimizer.

        kwargs: Hyperparameters for the training (passed as keyword arguments). There are the following hyperparameters:

            nsteps (int) : Number of training steps.

            optim (pennylane.optimize instance): Optimizer used during the training of the circuit.
                Pass as qml.OptimizerName.

            Tmax (list): Maximum point T as defined in https://arxiv.org/abs/2010.08512. (Definition 8)
                    The first element is the maximum number of parameters among all architectures,
                    the second is the maximum inference time among all architectures in terms of computing time,
                    the third one is the maximum inference time among all architectures in terms of the number of CNOTS
                    in the circuit
        
            rate_type (string): Determines the type of error rate in the W-coefficient.
                    If rate_type == 'accuracy', the inference time of the circuit
                    is equal to the time it takes to evaluate the accuracy of the trained circuit with
                    respect to a validation batch three times the size of the training batch size and 
                    the error rate is equal to 1-accuracy (w.r.t. to a validation batch).

                    If rate_type == 'accuracy', the inference time of the circuit is equal to the time
                    it takes to train the circuit (for nsteps training steps) and compute the cost at
                    each step and the error rate is equal to the cost after nsteps training steps.

            


 

    Returns:
        (W_,weights): W-coefficient, trained weights
    """
    print('batch_size',type(batch_size))
    # fix the seed while debugging
    #np.random.seed(1337)
    def ohe_cost_fcn(params, circuit, ang_array, actual):
        '''
        use MAE to start
        '''
        predictions = (np.stack([circuit(params, x) for x in ang_array]) + 1) * 0.5
        return mse(actual, predictions)

    def wn_cost_fcn(params, circuit, ang_array, actual):
        '''
        use MAE to start
        '''
        w = params[:,-1]

        theta = params[:,:-1]
        #print(w.shape,w,theta.shape,theta)
        predictions = np.asarray([2.*(1.0/(1.0+exp(np.dot(-w,circuit(theta, features=x)))))- 1. for x in ang_array])
        return mse(actual, predictions)

    if kwargs['readout_layer']=='one_hot':
        var = np.zeros(parameter_shape)
    elif kwargs['readout_layer']=="weighted_neuron":
        var = np.hstack((np.zeros(parameter_shape),np.random.random((kwargs['nqubits'],1))-0.5))
    rate_type = kwargs['rate_type']
    inf_time = kwargs['inf_time']
    optim = kwargs['optim']
    numcnots = kwargs['numcnots']

    Tmax = kwargs['Tmax'] #Tmax[0] is maximum parameter size, Tmax[1] maximum inftime (timeit),Tmax[2] maximum number of entangling gates
    num_train = len(Y_train)
    validation_size = int(0.1*num_train)
    opt = optim(stepsize=learning_rate) #all optimizers in autograd module take in argument stepsize, so this works for all
    start = time.time()
    for _ in range(kwargs['nsteps']):
        batch_index = np.random.randint(0, num_train, (batch_size,))
        X_train_batch = np.asarray(X_train[batch_index])
        Y_train_batch = np.asarray(Y_train[batch_index])
        if kwargs['readout_layer']=='one_hot':
            var, cost = opt.step_and_cost(lambda v: ohe_cost_fcn(v, circuit, X_train_batch, Y_train_batch), var)
        elif kwargs['readout_layer']=='weighted_neuron':
            var, cost = opt.step_and_cost(lambda v: wn_cost_fcn(v, circuit, X_train_batch, Y_train_batch), var)
    end = time.time()
    cost_time = (end - start)

    if kwargs['rate_type'] == 'accuracy':
        validation_batch = np.random.randint(0, num_train, (validation_size,))
        X_validation_batch = np.asarray(X_train[validation_batch])
        Y_validation_batch = np.asarray(Y_train[validation_batch])
        start = time.time()  # add in timeit function from Wbranch
        if kwargs['readout_layer']=='one_hot':
            predictions = np.stack([circuit(var, x) for x in X_validation_batch])
        elif kwargs['readout_layer']=='weighted_neuron':
            n = kwargs.get('nqubits')
            w = var[:,-1]
            theta = var[:,:-1]
            predictions = [int(np.round(2.*(1.0/(1.0+exp(np.dot(-w,circuit(theta, features=x)))))- 1.,1)) for x in X_validation_batch]
        end = time.time()
        inftime = (end - start) / len(X_validation_batch)
        if kwargs['readout_layer']=='one_hot':
            err_rate = (1.0 - ohe_accuracy(Y_validation_batch,predictions))+10**-7 #add small epsilon to prevent divide by 0 errors
            print('error rate:',err_rate)
            #print('weights: ',var)
        elif kwargs['readout_layer']=='weighted_neuron':
            err_rate = (1.0 - wn_accuracy(Y_validation_batch,predictions))+10**-7 #add small epsilon to prevent divide by 0 errors
            print('error rate:',err_rate)
            #print('weights: ',var)
    elif kwargs['rate_type'] == 'batch_cost':
        err_rate = (cost) + 10**-7 #add small epsilon to prevent divide by 0 errors
        print('error rate:',err_rate)
        #print('weights: ',var)
        inftime = cost_time
    # QHACK #

    if kwargs['inf_time'] =='timeit':

        W_ = np.abs((Tmax[0] - len(var)) / (Tmax[0])) * np.abs((Tmax[1] - inftime) / (Tmax[1])) * (1. / err_rate)

    elif kwargs['inf_time']=='numcnots':
        nc_ = numcnots
        W_ = np.abs((Tmax[0] - len(var)) / (Tmax[0])) * np.abs((Tmax[2] - nc_) / (Tmax[2])) * (1. / err_rate)

    return W_,var

def evaluate_w(circuit, n_params, X_train, Y_train, **kwargs):
    """
    together with the function train_circuit(...) this executes lines 7-8 in the Algorithm 1 pseudo code of (de Wynter 2020)
    batch_sets and learning_rates are lists, if just single values needed then pass length-1 lists
    """
    Wmax = 0.0
    batch_sets = kwargs.get('batch_sizes')
    learning_rates=kwargs.get('learning_rates')
    hyperparameter_space = list(itertools.product(batch_sets, learning_rates))
    for idx, sdx in hyperparameter_space:
        wtemp, weights = train_circuit(circuit, n_params,X_train, Y_train, batch_size=idx, learning_rate=sdx, **kwargs)
        if wtemp >= Wmax:
            Wmax = wtemp
            saved_weights = weights
    return Wmax, saved_weights
