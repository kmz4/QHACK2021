import pennylane as qml
import numpy as np #or import from pennylane?
#from pennylane import numpy as np
from torch.autograd import Variable
import torch
import time
import tensorflow as tf

from functools import partial



num_of_CNOTS_per_wire = [2,0,0]
nqubits = 2
expvalnum = 0


archtest = ['ZZ','X']
depthtest= len(archtest)
expvalnumtest=0
optimtest = torch.optim.Adam
stepstest = 20


def zz_layer(wires, params):
    nq = len(wires)
    for n in range(nq - 1):
        zz_gate([n, n + 1], params[n])
    zz_gate([nq - 1, 0], params[nq - 1])


def zz_gate(wires, gamma):
    qml.CNOT(wires=wires)
    qml.RZ(gamma, wires=wires[1])
    qml.CNOT(wires=wires)


def x_layer(wires, params):
    nqubits = len(wires)
    for n in range(nqubits):
        qml.RX(params[n], wires=[n, ])


def y_layer(wires, params):
    nqubits = len(wires)
    for n in range(nqubits):
        qml.RY(params[n], wires=[n, ])

#num_2_string_dict = {0: 'ZZ', 1: 'X', 2: 'Y'}
layer_language = {'ZZ': zz_layer, 'X': x_layer, 'Y': y_layer}

def num_of_entangling_gate(architecture,nwires):  #architecture has to be a list,not iterobject

    return architecture.count('ZZ')*2*nwires

def circuit_from_architecture(params,nqubits,architecture):
    it = 0
    for component in architecture:
            layer_language[component](list(range(nqubits)), params[:, it])
            it+=1
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))


 
""" def training(opt,nqubits,depth,steps,arch,expvalnum,cost_function):
    params_tr = Variable(torch.tensor(np.ones((nqubits, depth))),requires_grad=True)

    print(opt)

    opt = opt([params_tr],lr=0.01)



    for k in range(steps):

        opt.zero_grad()
        loss = cost_function(params_tr,nqubits,expvalnum,arch)
        loss.backward()

         
         
        opt.step()
    return loss """

def some_loss_func_torch(y):

    target = 1
    return torch.abs(target-y)**2

def some_loss_func_tf(y):

    target = 0.5

    return tf.abs(target-y)**2

def some_loss_func(y):

    target = 0.5

    return np.abs(target-y)**2

def Wfunc(arch,nqu,Tmax,loss_func,steps,opt,optimoptions,interface = None,backend='default.qubit',time_measure='timeit'): 
#arch must be list, right now interface has to be torch
#Tmax[0]= maximum param number #Tmax[1] = maximum time with time it #Tmax[2] = maximum number of entangling gates
    depth = len(arch) 

    dev = qml.device(backend, wires=nqu)

    circuit = qml.QNode(circuit_from_architecture, dev,interface = interface)

    numparam = nqu*depth

    def cost_function(params,arch):
        
        
        if type(params).__name__ == 'ArrayBox':
             
            params1 = params._value
        else: 
            params1 = params

        y = circuit(params1,nqubits,arch)[expvalnum]

         
         

        return loss_func(y)

    if interface==None:
 
        params_tr =  np.ones((nqu, depth),requires_grad=True) #does this need requires grad?
        opt = opt(**optimoptions)

    if interface == 'torch':

        params_tr = Variable(torch.tensor(np.ones((nqubits, depth))),requires_grad=True)
        opt = opt([params_tr],**optimoptions)

    if interface == 'tf':

        params_tr = tf.Variable(np.ones((nqu, depth)))
        opt = opt(**optimoptions)


    

    start = time.time()
   
 
 

    if interface=='torch':
        for k in range(steps):

            opt.zero_grad()
            
            loss = cost_function(params_tr,arch)
            loss.backward()
            print(loss)
            opt.step()
        

    elif interface==None:
         
        for k in range(steps):

            #loss = cost_function(params_tr,arch)
            loss = cost_function(params_tr,arch)
            #print('loss', loss)
            params_tr = opt.step(partial(cost_function, arch=arch), params_tr)
             


             

    elif interface=='tf':

        

        opt.minimize(partial(cost_function, arch=arch), var_list=[params])


        #for k in range(steps):

            #with tf.GradientTape() as tape:

                
                 

            #gradients = tape.gradient(loss, params_tr) #honestly i just copied this from a tutorial page, 
            #am not familiar with this tape function
             
            #print(type(params_tr))
            #opt.apply_gradients(zip(gradients, params_tr))
            

    loss = float(loss) #not sure if this works for all interfaces
    end = time.time()
    inftime = end-start

    if time_measure=='timeit':

        print('timeit')

        W = (Tmax[0]-numparam*(Tmax[1]-inftime))/(Tmax[0]*Tmax[1]*loss)

    if time_measure == 'entgates':

        numentgates = num_of_entangling_gate(arch,nqu)

        W = (Tmax[0]-numparam*(Tmax[2]-inftime))/(Tmax[0]*Tmax[2]*loss)


    return W




#print(Wfunc(archtest,2,[100,100,100],some_loss_func_torch,100,torch.optim.Adam,{'lr':0.1},interface='torch'))
print(Wfunc(archtest,2,[100,100,100],some_loss_func_tf,100,tf.keras.optimizers.Adam,{'learning_rate':0.1},interface='tf'))

#print(Wfunc(archtest,2,[100,100,100],some_loss_func,20,qml.optimize.gradient_descent.GradientDescentOptimizer,{'stepsize':0.05}))     
  





""" start = time.time()

lossatend = training(optimtest,nqubits,depthtest,stepstest,archtest,0)
end = time.time()
inftime = end-start
print(end-start)

print(num_of_entangling_gate(archtest,nqubits)) """


        

