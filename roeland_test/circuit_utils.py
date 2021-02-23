import pennylane as qml


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