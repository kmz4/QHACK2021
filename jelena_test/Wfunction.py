import pennylane as qml
import numpy as np
from torch.autograd import Variable
import torch
import time

num_2_string_dict = {0: 'ZZ', 1: 'X', 2: 'Y'}
layer_language = {'ZZ': zz_layer, 'X': x_layer, 'Y': y_layer}

num_of_CNOTS_per_wire = [2,0,0]

dev = qml.device("default.qubit", wires=nqubits)


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

def num_of_entangling_gate(architecture,nwires): #takes in architecture 'object' as Roeland has defined it #architecture has to be a list

    return architecture.count(0)*2*nwires

def circuit_from_architecture(params,nqubits):
    it = 0
    for component in architecture:
            layer_language[component](list(range(nqubits)), params[:, it])
            it+=1
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

circuit = qml.QNode(circuit_from_architecture, dev,interface=torch)

def cost_function(params,nqubits,expvalnum):

    target = 0.5

    return torch.abs(target-circuit(params,nqubits)[expvalnum])**2


optim = torch.optim.Adam

def training(cost,opt,nqubits,depth,steps,opt):
    params_tr = Variable(torch.tensor(np.ones((nqubits, depth))),requires_grad=True)

    opt([params_tr],lr=0.01)

    for k in range(steps):

        opt.zero_grad()
        loss = cost_function(params_tr,nqubits,expvalnum)
        loss.backward()
         
        opt.step()
    return loss

start = time.time()

print(hello)
end = time.time()
print(start-end)


        

