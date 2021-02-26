import pennylane as qml
from pennylane import numpy as np
from qose.featuremaps import *


def cycle_CNOT_layer(wires):
    """

    Args:
      wires: 

    Returns:

    """
    nq = len(wires)
    for n in range(nq - 1):
        qml.CNOT(wires=[n, n + 1])
    qml.CNOT(wires=[nq - 1, 0])


def path_CNOT_layer(wires):
    """

    Args:
      wires: 

    Returns:

    """
    nq = len(wires)
    for n in range(nq - 1):
        qml.CNOT(wires=[n, n + 1])


def zz_layer(wires, params):
    """

    Args:
      wires: 
      params: 

    Returns:

    """
    nq = len(wires)
    for n in range(nq - 1):
        zz_gate([n, n + 1], params[n])
    zz_gate([nq - 1, 0], params[nq - 1])


def zz_gate(wires, gamma):
    """

    Args:
      wires: 
      gamma: 

    Returns:

    """
    qml.CNOT(wires=wires)
    qml.RZ(gamma, wires=wires[1])
    qml.CNOT(wires=wires)


def x_layer(wires, params):
    """

    Args:
      wires: 
      params: 

    Returns:

    """
    nqubits = len(wires)
    for n in range(nqubits):
        qml.RX(params[n], wires=[n, ])


def z_layer(wires, params):
    """

    Args:
      wires: 
      params: 

    Returns:

    """
    nqubits = len(wires)
    for n in range(nqubits):
        qml.RZ(params[n], wires=[n, ])


def y_layer(wires, params):
    """

    Args:
      wires: 
      params: 

    Returns:

    """
    nqubits = len(wires)
    for n in range(nqubits):
        qml.RY(params[n], wires=[n, ])


def embedding_1(X, wires, fill='redundant'):
    """

    Args:
      X: 
      wires: 
      fill:  (Default value = 'redundant')

    Returns:

    """
    if len(X) < len(wires):
        r_ = len(wires) // len(X)
        if fill == 'redundant':
            large_features = np.tile(X, r_)
        elif fill == 'pad':
            large_features = np.pad(X, (0, len(wires)), 'constant', constant_values=0)
        qml.templates.embeddings.AngleEmbedding(large_features, wires=wires, rotation='Y')
    else:
        qml.templates.embeddings.AngleEmbedding(X, wires=wires, rotation='Y')


def embedding_2(X, wires):
    """

    Args:
      X: 
      wires: 

    Returns:

    """
    qml.templates.embeddings.AmplitudeEmbedding(X, wires=wires, pad=0, normalize=True)

def construct_circuit_from_leaf(leaf: str, nqubits: int, nclasses: int, dev: qml.Device, config: dict):
    """Construct a Qnode specified by the architecture in the leaf. This includes an embedding layer as first layer.

    Args:
      leaf: String that corresponds to a leaf in the tree.
      nqubits: The number of qubits in the circuit.
      nclasses: The number of classes in the circuit.
      dev: PennyLane Device.
      leaf: str: 
      nqubits: int: 
      nclasses: int: 
      dev: qml.Device: 
      config: dict: 

    Returns:
      QNode corresponding to the circuit.

    """
    architecture = leaf.split(':')

    # The first layer must be an embedding layer
    embedding_circuit = architecture.pop(0)

    def circuit_from_architecture(params, features):
        """

        Args:
          params: 
          features: 

        Returns:

        """
        # lookup the function in the embedding dict, call with features being passed.
        string_to_embedding_mapping[embedding_circuit](features, dev.wires)
        # for each layer, lookup the function in the layer dict, call with parameters being passed.
        for d, component in enumerate(architecture):
            if component == 'hw_CNOT':
                string_to_layer_mapping[component](list(range(nqubits)))
            else:
                string_to_layer_mapping[component](list(range(nqubits)), params[:, d])
        # return an expectation value for each class so we can compare with one hot encoded labels.
        return [qml.expval(qml.PauliZ(nc)) for nc in range(nclasses)]

    # Return the shape of the parameters so we can initialize them correctly later on.
    if config['circuit_type'] == 'schuld':
        params_shape = (nqubits, len(architecture))
        numcnots = architecture.count('ZZ') * 2 * nqubits  # count the number of cnots
    elif config['circuit_type'] == 'hardware':
        param_gates = [x for x in architecture if x != 'CNOT']
        params_shape = (nqubits, len(param_gates))
        params_shape = (nqubits, len(architecture))
        numcnots = architecture.count('CNOT') * (nqubits - 1)  # Each CNOT layer has (n-1) CNOT gates
    # create and return a QNode
    return qml.QNode(circuit_from_architecture, dev), params_shape, numcnots  # give back the number of cnots

string_to_layer_mapping = {'ZZ': zz_layer, \
                           'X': x_layer, 'Y': y_layer, 'Z': z_layer, \
                           'hw_CNOT': path_CNOT_layer}
string_to_embedding_mapping = {'E1': embedding_1, 'E2': embedding_2, 'E3': qaoa, 'E4': XXZ, 'E5': aspuru,
                               'E6': random_embed}
