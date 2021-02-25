import pennylane as qml
from pennylane import numpy as np

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

def embedding_1(X, wires):
    if len(X)<len(wires):
        r_ = len(wires)//len(X)
        large_features = np.tile(X,r_)
        qml.templates.embeddings.AngleEmbedding(large_features, wires=wires, rotation='Y') # replace with more general embedding
    else:
        qml.templates.embeddings.AngleEmbedding(X, wires=wires, rotation='Y') # replace with more general embedding        
    qml.templates.embeddings.AngleEmbedding(X, wires=wires)


# TODO: ADD W-COST HERE

string_to_layer_mapping = {'ZZ': zz_layer, 'X': x_layer, 'Y': y_layer}
string_to_embedding_mapping = {'E1': embedding_1}
