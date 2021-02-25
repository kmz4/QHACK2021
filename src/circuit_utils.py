import pennylane as qml
from pennylane import numpy as np

def cycle_CNOT_layer(wires):
    nq = len(wires)
    for n in range(nq-1):
        CNOT(wires=[n,n+1])
    CNOT(wires=[nq-1,0])

def path_CNOT_layer(wires):
    nq = len(wires)
    for n in range(nq-1):
        CNOT(wires=[n,n+1])

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

def z_layer(wires, params):
    nqubits = len(wires)
    for n in range(nqubits):
        qml.RZ(params[n], wires=[n, ])

def y_layer(wires, params):
    nqubits = len(wires)
    for n in range(nqubits):
        qml.RY(params[n], wires=[n, ])

def embedding_1(X, wires,fill='redundant'):
    if len(X)<len(wires):
        r_ = len(wires)//len(X)
        if fill=='redundant':
            large_features = np.tile(X,r_)
        elif fill=='pad':
            large_features = np.pad(X,(0,len(wires)),'constant',constant_values=0)
        qml.templates.embeddings.AngleEmbedding(large_features, wires=wires, rotation='Y') # replace with more general embedding
    else:
        qml.templates.embeddings.AngleEmbedding(X, wires=wires, rotation='Y') # replace with more general embedding
    qml.templates.embeddings.AngleEmbedding(X, wires=wires)


# TODO: ADD W-COST HERE

string_to_layer_mapping = {'ZZ': zz_layer,\
                        'X': x_layer, 'Y': y_layer,'Z':z_layer,\
                        'hw_CNOT':path_CNOT_layer}
string_to_embedding_mapping = {'E1': embedding_1}
