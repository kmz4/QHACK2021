"""
Feature maps
************
This module contains feature maps. Each feature map function
takes an input vector x and weights, and constructs a circuit that maps
these two to a quantum state. The feature map function can be called in a qnode.
A feature map has the following positional arguments: weights, x, wires. It can have optional
keyword arguments.
Each feature map comes with a function that generates initial parameters
for that particular feature map.
The function get_embedding_info can be used to get information about the number of
parameters and  gates for any of the embeddings
"""
import numpy as np
import pennylane as qml


def _entanglerZ(w, wire1, wire2):
    qml.CNOT(wires=[wire2, wire1])
    qml.RZ(2 * w, wires=wire1)
    qml.CNOT(wires=[wire2, wire1])


def qaoa(weights, x, wires, n_layers=1):
    """
  1-d Ising-coupling QAOA feature map, according to arXiv1812.11075.
  Example one layer, 4 wires, 2 inputs:

     |0> - R_x(x1) - |^| -------- |_| - R_y(w7)  -
     |0> - R_x(x2) - |_|-|^| ---------- R_y(w8)  -
     |0> - ___H___ ------|_|-|^| ------ R_y(w9)  -
     |0> - ___H___ ----------|_| -|^| - R_y(w10) -

  After the last layer, another block of R_x(x_i) rotations is applied.

  :param weights: trainable weights of shape 2*n_layers*n_wires
  :param x: input, len(x) is <= len(wires)
  :param wires: list of wires on which the feature map acts
  :param n_layers: number of repetitions of the first layer
  """
    n_wires = len(wires)

    if n_wires == 1:
        n_weights_needed = n_layers
    else:
        n_weights_needed = 2 * n_wires * n_layers

    if len(x) > n_wires:
        raise ValueError("Feat map can encode at most {} features (which is the "
                         "number of wires), got {}.".format(n_wires, len(x)))

    if len(weights) != n_weights_needed:
        raise ValueError("Feat map needs {} weights, got {}."
                         .format(n_weights_needed, len(weights)))

    for l in range(n_layers):

        # inputs
        for i in range(n_wires):
            # Either feed in feature
            if i < len(x):
                qml.RX(x[i], wires=wires[i])
            # or a Hadamard
            else:
                qml.Hadamard(wires=wires[i])

        # 1-d nearest neighbour coupling
        if n_wires == 1:
            qml.RY(weights[l], wires=wires[0])

        else:
            for i in range(n_wires):
                if i < n_wires - 1:
                    _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[i + 1])
                else:
                    # enforce periodic boundary condition
                    _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[0])
            # local fields
            for i in range(n_wires):
                qml.RY(weights[l * 2 * n_wires + n_wires + i], wires=wires[i])

    # repeat feature encoding once more at the end
    for i in range(n_wires):
        # Either feed in feature
        if i < len(x):
            qml.RX(x[i], wires=wires[i])
        # or a Hadamard
        else:
            qml.Hadamard(wires=wires[i])


def pars_qaoa(n_wires, n_layers=1):
    """
  Initial weight generator for 1-d qaoa feature map
  :param n_wires: number of wires
  :param n_layers: number of layers
  :return: array of weights
  """
    if n_wires == 1:
        return 0.001 * np.ones(n_layers)
    else:
        return 0.001 * np.ones(n_wires * n_layers * 2)




def XXZ(w_zz, w_rot, x, wires, n_layers=1):
    """
  :param w_zz: trainable weights for ZZ gates
  :param w_rot: trainable weights for RX gates
  :param x: input, len(x) is <= len(wires)
  :param wires: list of wires on which the feature map acts
  :param n_layers: number of repetitions of the first layer
  """
    n_wires = len(wires)

    if n_wires == 1:
        raise ValueError("use at least 2 wires to enable entangling gates")
    else:
        n_weights_needed = (n_wires + n_wires - len(x)) * n_layers

    if len(x) > n_wires:
        raise ValueError("Feat map can encode at most {} features (which is the "
                         "number of wires), got {}.".format(n_wires, len(x)))
    l_total = len(w_zz) + len(w_rot)
    if l_total != n_weights_needed:
        raise ValueError("Feat map needs {} number of weights, got {}."
                         .format(n_weights_needed, l_total))

    for l in range(n_layers):

        # hadamards
        for i in range(n_wires):
            qml.Hadamard(wires=wires[i])

        # nearest neighbour coupling

        for i in range(n_wires):
            if i < n_wires - 1:
                _entanglerZ(w_zz[l * n_wires + i], wires[i], wires[i + 1])
            else:
                # enforce periodic boundary condition
                _entanglerZ(w_zz[l * n_wires + i], wires[i], wires[0])

        # feature encoding
        for i in range(n_wires):
            # Either feed in feature
            if i < len(x):
                qml.RX(x[i], wires=wires[i])
            # or a Hadamard
            else:
                qml.RX(w_rot[i-l], wires=wires[i])

def pars_xxz(x, n_wires, n_layers=1):
    """
  Initial weight generator for xxz feature map
  :param n_wires: number of wires
  :param n_layers: number of layers
  :return: array of weights
  """
    w_zz = 0.001 * np.ones(n_wires * n_layers)
    w_rot = 0.001 * np.ones((n_wires - len(x))*n_layers)
    return w_zz, w_rot



def aspuru(weights, x, wires, n_layers=1):
    """ Circuits ID = 5 in arXiv:1905.10876 paper
  :param weights: trainable weights
  :param x: input, len(x) is <= len(wires)
  :param wires: list of wires on which the feature map acts
  :param n_layers: number of repetitions of the first layer
  """
    data_size = len(x)
    n_wires = len(wires)
    weights_each_layer = (n_wires * (n_wires + 3) - 2 * data_size)
    n_weights_needed = weights_each_layer * n_layers

    if len(x) > n_wires:
        raise ValueError("Feat map can encode at most {} features (which is the "
                         "number of wires), got {}.".format(n_wires, len(x)))

    if len(weights) != n_weights_needed:
        raise ValueError("Feat map needs {} weights, got {}."
                         .format(n_weights_needed, len(weights)))

    for l in range(n_layers):

        # inputs
        for i in range(data_size):
            if i < len(x):
                qml.RX(x[i], wires=wires[i])

        for i in range(len(x), n_wires):
            qml.RX(weights[weights_each_layer * l + i -data_size], wires=wires[i])

        for i in range(n_wires):
            qml.RZ(weights[weights_each_layer * l + n_wires - data_size + i], wires=wires[i])

        for i in reversed(range(n_wires)):
            for j in reversed(range(n_wires)):
                if j == i:
                    continue

                qml.CRZ(weights[weights_each_layer * l + 2 * n_wires - data_size + i * (n_wires - 1) + j],
                            wires=[wires[i], wires[j]])

        for i in range(data_size):
            qml.RX(x[i], wires=wires[i])

        for i in range(len(x), n_wires):
            qml.RX(weights[weights_each_layer * l + n_wires * (n_wires + 1) - data_size + i],
                   wires=wires[i])

        for i in range(n_wires):
            qml.RZ(weights[weights_each_layer * l + n_wires * (n_wires + 2) - 2 * data_size + i], wires=wires[i])


def pars_aspuru(x, n_wires, n_layers=1):
    weights_each_layer = (n_wires * (n_wires + 3) - 2 * len(x))

    return 0.001 * np.ones(n_layers * weights_each_layer)

def random_embed(weights, x, wires, n_layers=1):
  """ random enbedding circuit
  :param weights: trainable weights
  :param x: input, len(x) is <= len(wires)
  :param wires: list of wires on which the feature map acts
  :param n_layers: number of repetitions of the first layer
  """
  n_wires = len(wires)
  n_weights_needed = n_layers * n_wires
  gate_set = [qml.RX, qml.RY, qml.RZ]
  for l in range(n_layers):
      i=0
      while i< len(x):
          gate = np.random.choice(gate_set)
          gate(x[i], wires=wires[i])
          i = i+1
      for i in range(n_wires):
          gate = np.random.choice(gate_set)
          gate(weights[l*n_wires+i -n_wires], wires=wires[i])
      qml.broadcast(qml.CNOT, wires = range(n_wires), pattern = "ring")

def pars_random(x, n_wires, n_layers=1):
    weights_each_layer = (n_wires * (n_wires + 3) - 2 * len(x))

    return 0.001 * np.ones(n_layers * n_wires)

def get_embedding_info(name, x, n_wires, n_layers):
    """
  get information about the number of weights, entangling gates, number of gates with
  input/non-trainable parameters and number of non-parametrized gates (hadamards) in an embedding
  :param name: name of embedding used
  :param x: input, len(x) is <= len(wires)
  :param n_wires: number of wires used in the architecture
  :param n_layers: number of layers used in the architecture
  """
    rem = n_wires - len(x)
    if name == "qaoa":
        return 2 * n_wires * n_layers, n_wires*n_layers, (n_layers * len(x))+len(x), (n_layers * rem)+rem
    if name == "xxz":
        return (n_wires + n_wires - len(x)) * n_layers, n_wires*n_layers, len(x) * n_layers, n_wires*n_layers
    if name == "aspuru":
        return n_layers* (n_wires * (n_wires + 3) - 2 * len(x)), (n_wires-1)*n_wires, len(x)*2*n_layers, 0
    if name == "angle":
        return 0, 0, len(x), 0
    if name == "amplitude":
        return 0, 0, len(x), 0
    if name == "random":
        return n_layers * n_wires, n_layers * n_wires, n_layers * len(x), 0

