import pennylane as qml
import numpy as np
from torch.autograd import Variable
import torch
import time



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

num_2_string_dict = {0: 'ZZ', 1: 'X', 2: 'Y'}
layer_language = {'ZZ': zz_layer, 'X': x_layer, 'Y': y_layer}

def num_of_entangling_gate(architecture,nwires): #takes in architecture 'object' as Roeland has defined it #architecture has to be a list

    return architecture.count('ZZ')*2*nwires

def circuit_from_architecture(params,nqubits,architecture):
    it = 0
    for component in architecture:
            layer_language[component](list(range(nqubits)), params[:, it])
            it+=1
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))



def cost_function(circuit,params,nqubits,expvalnum,arch):

    target = 0.5

    return torch.abs(target-circuit(params,nqubits,arch)[expvalnum])**2




def training(opt,nqubits,depth,steps,arch,expvalnum,cost_function):
    params_tr = Variable(torch.tensor(np.ones((nqubits, depth))),requires_grad=True)

    print(opt)

    opt = opt([params_tr],lr=0.01)



    for k in range(steps):

        opt.zero_grad()
        loss = cost_function(params_tr,nqubits,expvalnum,arch)
        loss.backward()

         
         
        opt.step()
    return loss

def some_loss_func(y):

    target = 0.5

    return torch.abs(target-y)**2

def Wfunc(arch,nqu,Tmax,time_measure='timeit',interface,backend,loss_func,steps,**optimoptions): #arch must be list, #right now interface has to be torch

    depth = len(arch) 

    dev = qml.device(backend, wires=nqu)

    circuit = qml.QNode(circuit_from_architecture, dev,interface = interface)

    numparam = nqu*depth

    def cost_function(params,arch):

        y = circuit(params,nqubits,arch)[expvalnum]

        return loss_func(y)

    params_tr = Variable(torch.tensor(np.ones((nqu, depth))),requires_grad=True)

    opt = opt([params_tr],**optimoptions)

    start = time.time()

    for k in range(steps):

        opt.zero_grad()
        loss = cost_function(params_tr,arch)
        loss.backward()

         
         
        opt.step()

    end = time.time()
    inftime = end-start

    if time_measure=='timeit':


     
    





start = time.time()

lossatend = training(optimtest,nqubits,depthtest,stepstest,archtest,0)
end = time.time()
inftime = end-start
print(end-start)

print(num_of_entangling_gate(archtest,nqubits))


        

