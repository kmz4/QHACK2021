import pennylane as qml
from pennylane import numpy as np

import networkx as nx

from sklearn import datasets
from typing import List

from circuit_utils import string_to_layer_mapping, string_to_embedding_mapping
from train_utils import train_circuit
from plot_utils import plot_tree


def tree_prune(G: nx.DiGraph, leaves_at_depth_d: dict, d: int, prune_rate: float):
    """
    Remove nodes from the tree based on the set prune rate and the total cost of the path from root to leaf.

    :param G: NetworkX DiGraph object that represents our tree.
    :param leaves_at_depth_d: Dictonary that keeps track of all the leaves at level d
    :param d: the depth that we are pruning at
    :param prune_rate: The percentage of leaves to be removed

    """
    cost_at_leaf = []
    # loop over the leaves at depth d
    for leaf in leaves_at_depth_d[d - 1]:
        cost_at_leaf.append(tree_cost_of_path(G, leaf))
    # Sort both leaves and cost according to cost, ascendingly
    leaves_sorted = [x for _, x in sorted(zip(cost_at_leaf,
                                              leaves_at_depth_d[d - 1]))]
    leaves_kept = leaves_sorted[int(np.ceil(prune_rate * len(cost_at_leaf))):]
    leaves_removed = leaves_sorted[:int(np.ceil(prune_rate * len(cost_at_leaf)))]
    G.remove_nodes_from(leaves_removed)
    leaves_at_depth_d[d - 1] = leaves_kept


def tree_grow_root(G: nx.DiGraph, leaves_at_depth_d: dict, layers: List[str]):
    """
    Initialize the tree with edges from the Root to the first branches.

    :param G: NetworkX DiGraph object that represents our tree.
    :param leaves_at_depth_d: Dictonary that keeps track of all the leaves at level d
    :param layers: List of strings containing embedding layers that can be added as first layer.

    """
    # loop over the possible layers that we can add
    for architecture in layers:
        G.add_edge('ROOT', architecture)
        leaves_at_depth_d[1].append(architecture)


def tree_grow(G: nx.DiGraph, leaves_at_depth_d: dict, d: int, layers: List[str]):
    """
    Grow the tree by adding a circuit layer.

    :param G: NetworkX DiGraph object that represents our tree.
    :param leaves_at_depth_d: Dictonary that keeps track of all the leaves at level d
    :param d: the depth that we are pruning at
    :param layers: List of strings of possible classification layers that can be added.

    """
    # loop over the leaves at depth d
    for architecture in leaves_at_depth_d[d - 1]:
        # for each leaf, add each of the possible layers to it in new leaves.
        for new_layer in layers:
            new_architecture = ':'.join([architecture, new_layer])
            comm_check = new_architecture.split(':')
            # Check if the layer we want to add is not the same as the previous one
            if comm_check[-2] != comm_check[-1]:
                G.add_node(new_architecture)
                G.add_edge(architecture, new_architecture)
                leaves_at_depth_d[d].append(new_architecture)


def tree_cost_of_path(G: nx.DiGraph, leaf: str) -> float:
    """
    Calculate the cost of going from the root of the tree to a leaf. Total cost is the sum of all W-costs.

    :param G: NetworkX DiGraph object that represents our tree.
    :param leaf: String that corresponds to a leaf in the tree.
    :return: float value corresponding to the cost.
    """
    paths = nx.shortest_path(G, 'ROOT', leaf)
    return sum([G.nodes[node]['W'] for node in paths])


def construct_circuit_from_leaf(leaf: str, nqubits: int, nclasses: int, dev: qml.Device):
    """
    Construct a Qnode specified by the architecture in the leaf. This includes an embedding layer as first layer.

    :param leaf: String that corresponds to a leaf in the tree.
    :param nqubits: The number of qubits in the circuit.
    :param nclasses:  The number of classes in the circuit.
    :param dev: PennyLane Device.
    :return: QNode corresponding to the circuit.
    """
    architecture = leaf.split(':')

    # The first layer must be an embedding layer
    embedding_circuit = architecture.pop(0)

    def circuit_from_architecture(params, features):
        # lookup the function in the embedding dict, call with features being passed.
        string_to_embedding_mapping[embedding_circuit](features, dev.wires)
        # for each layer, lookup the function in the layer dict, call with parameters being passed.
        for d, component in enumerate(architecture):
            string_to_layer_mapping[component](list(range(nqubits)), params[:, d])
        # return an expectation value for each class so we can compare with one hot encoded labels.
        return [qml.expval(qml.PauliZ(nc)) for nc in range(nclasses)]

    # Return the shape of the parameters so we can initialize them correctly later on.
    params_shape = (nqubits, len(architecture))
    # create and return a QNode
    return qml.QNode(circuit_from_architecture, dev), params_shape


def run_tree_architecture_search(config: dict, dev_type: str):
    """
    The main workhorse for running the algorithm

    dev_type: device type specified as either "remote" or "local"
    "remote" is to use aws SV1
    "local" is to use pennylane simulator

    :param config: Dictionary with configuration parameters for the algorithm. Possible keys are:
        - nqubits: Integer. The number of qubits in the circuit
        - min_tree_depth: Integer. Minimum circuit depth before we start pruning
        - max_tree_depth: Integer. Maximum circuit depth
        - prune_rate: Integer. Percentage of nodes that we throw away when we prune
        - prune_step: Integer. How often do we prune
        - plot_trees: Boolean. Do we want to plot the tree at every depth?
        - data_set: String. Which dataset are we learning? Can be 'moons' or 'circles'
        - nsteps: Integer. The number of steps for training.
        - opt: qml.Optimizer. Pennylane optimizer
        - batch_size: Integer. Batch size for training.
        - n_samples: Integer. Number of samples that we want to take from the data set.
        - learning_rate: Float. Optimizer learning rate.
        - save_frequency: Integer. How often do we want to save the tree? Set to 0 for no saving.
        - save_path: String. Location to store the data.

    """
    # Parse configuration parameters.
    NQUBITS = config['nqubits']
    NSAMPLES = config['n_samples']
    PATH = config['save_path']

    if dev_type == "local":
        dev = qml.device("default.qubit.autograd", wires=NQUBITS)
    elif dev_type == "remote":
        my_bucket = "amazon-braket-0fc49b964f85"  # the name of the bucket
        my_prefix = PATH.split('/')[1]  # name of the folder in the bucket is the same as experiment name
        s3_folder = (my_bucket, my_prefix)
        device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
        dev = qml.device("braket.aws.qubit", device_arn=device_arn, wires=NQUBITS, s3_destination_folder=s3_folder, parallel=True, max_parallel=10, poll_timeout_seconds=30)

    MIN_TREE_DEPTH = config['min_tree_depth']
    MAX_TREE_DEPTH = config['max_tree_depth']
    SAVE_FREQUENCY = config['save_frequency']

    PRUNE_DEPTH_STEP = config['prune_step']  # EVERY ith step is a prune step
    PRUNE_RATE = config['prune_rate']  # Percentage of nodes to throw away at each layer
    PLOT_INTERMEDIATE_TREES = config['plot_trees']

    assert MIN_TREE_DEPTH < MAX_TREE_DEPTH, 'MIN_TREE_DEPTH must be smaller than MAX_TREE_DEPTH'
    assert 0.0 < PRUNE_RATE < 1.0, f'The PRUNE_RATE must be between 0 and 1, found {PRUNE_RATE}'

    if config['data_set'] == 'circles':
        X_train, y_train = datasets.make_circles(n_samples=NSAMPLES, factor=.5, noise=.05)
    elif config['data_set'] == 'moons':
        X_train, y_train = datasets.make_moons(n_samples=NSAMPLES, noise=.05)
    # rescale data to -1 1
    X_train = np.multiply(1.0, np.subtract(np.multiply(np.divide(np.subtract(X_train, X_train.min()),
                                                                 (X_train.max() - X_train.min())), 2.0), 1.0))
    # one hot encode labels
    y_train_ohe = np.zeros((y_train.size, y_train.max() + 1))
    y_train_ohe[np.arange(y_train.size), y_train] = 1
    # automatically determine the number of classes
    NCLASSES = len(np.unique(y_train))
    assert NQUBITS >= NCLASSES, 'The number of qubits must be equal or larger than the number of classes'

    # Create a directed graph.
    G = nx.DiGraph()
    # Add the root
    G.add_node("ROOT")
    nx.set_node_attributes(G, {'ROOT': 0.0}, 'W')
    # Define allowed layers
    possible_layers = ['ZZ', 'X', 'Y']
    possible_embeddings = ['E1', ]
    assert all([l in string_to_layer_mapping.keys() for l in possible_layers]), 'No valid mapping from string to function found'
    assert all([l in string_to_embedding_mapping.keys() for l in possible_embeddings]), 'No valid mapping from string to function found'
    leaves_at_depth_d = dict(zip(range(MAX_TREE_DEPTH), [[] for _ in range(MAX_TREE_DEPTH)]))
    leaves_at_depth_d[0].append('ROOT')
    # Iteratively construct tree, pruning at set rate
    for d in range(1, MAX_TREE_DEPTH):
        print(f"Depth = {d}")
        # Save trees
        if (SAVE_FREQUENCY > 0) & ~(d % SAVE_FREQUENCY):
            nx.write_gpickle(G, config['save_path'] + f'/tree_depth_{d}.pickle')
        # Plot trees
        if PLOT_INTERMEDIATE_TREES:
            plot_tree(G)
        # If we are not passed MIN_TREE_DEPTH, don't prune
        if d < MIN_TREE_DEPTH:
            # First depth connects to root
            if d == 1:
                tree_grow_root(G, leaves_at_depth_d, possible_embeddings)
                # At the embedding level we don't need to train because there are no params.
                for v in leaves_at_depth_d[d]:
                    nx.set_node_attributes(G, {v: 1.0}, 'W')
            else:
                tree_grow(G, leaves_at_depth_d, d, possible_layers)
                # For every leaf, create a circuit and run the optimization.
                for v in leaves_at_depth_d[d]:
                    print(f'Training leaf {v}')
                    circuit, pshape = construct_circuit_from_leaf(v, NQUBITS, NCLASSES, dev)
                    w_cost = train_circuit(circuit, pshape, X_train, y_train_ohe, 'accuracy', **config)
                    # Add the w_cost to the node so we can use it later for pruning
                    nx.set_node_attributes(G, {v: w_cost}, 'W')

        else:
            # Check that we are at the correct prune depth step.
            if not (d - MIN_TREE_DEPTH) % PRUNE_DEPTH_STEP:
                print('Prune Tree')
                tree_prune(G, leaves_at_depth_d, d, PRUNE_RATE)
                print('Grow Pruned Tree')
                tree_grow(G, leaves_at_depth_d, d, possible_layers)
                # For every leaf, create a circuit and run the optimization.
                for v in leaves_at_depth_d[d]:
                    print(f'Training leaf {v}')
                    circuit, pshape = construct_circuit_from_leaf(v, NQUBITS, NCLASSES, dev)
                    w_cost = train_circuit(circuit, pshape, X_train, y_train_ohe, 'accuracy', **config)
                    # Add the w_cost to the node so we can use it later for pruning
                    nx.set_node_attributes(G, {v: w_cost}, 'W')

            else:
                print('Grow Tree')
                tree_grow(G, leaves_at_depth_d, d, possible_layers)
                for v in leaves_at_depth_d[d]:
                    print(f'Training leaf {v}')
                    circuit, pshape = construct_circuit_from_leaf(v, NQUBITS, NCLASSES, dev)
                    w_cost = train_circuit(circuit, pshape, X_train, y_train_ohe, 'accuracy', **config)
                    nx.set_node_attributes(G, {v: w_cost}, 'W')
