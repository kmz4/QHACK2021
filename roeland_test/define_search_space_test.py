import pennylane as qml
import itertools as it
import numpy as np


## What possible layers can we add at depth d?

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


def non_commuting_check(architecture):
    for g1, g2 in zip(architecture[:-1], architecture[1:]):
        if g1 == g2:
            return False
    return True


## ZZ connections, Single X, Single Z

layer_language = {'ZZ': zz_layer, 'X': x_layer, 'Y': y_layer}
depth = 4
nqubits = 3
num_y_labels = 3
assert nqubits >= num_y_labels, f"nqubits must be larger than {nqubits}, since num_y_labels = {num_y_labels}"
dev = qml.device("default.qubit", wires=nqubits)

n_architecures = 0
for architecture in it.product(('ZZ', 'X', 'Y'), repeat=depth):
    print(architecture)
    init_params = np.ones((nqubits, depth))
    if non_commuting_check(architecture):
        n_architecures += 1


    def circuit_from_architecture(params):
        for d, component in enumerate(architecture):
            layer_language[component](list(range(nqubits)), params[:, d])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))


    circuit = qml.QNode(circuit_from_architecture, dev)
    print(circuit(init_params))
print(n_architecures)
## Monte Carlo Tree Search this space of possible architectures
## Start with identity circuit
