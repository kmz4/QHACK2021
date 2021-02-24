import pennylane as qml
from pennylane import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from sklearn import datasets
from circuit_utils import string_to_layer_mapping, string_to_embedding_mapping
from train_utils import train_circuit


def arbitrary_cost_fn():
    return np.random.rand(1)


def non_commuting_check(architecture):
    for g1, g2 in zip(architecture[:-1], architecture[1:]):
        if g1 == g2:
            return False
    return True


def tree_prune(G, leaves_at_depth_d, d, prune_rate):
    cost_at_leaf = []
    for leaf in leaves_at_depth_d[d - 1]:
        cost_at_leaf.append(tree_cost_of_path(G, leaf))

    leaves_sorted = [x for _, x in sorted(zip(cost_at_leaf,
                                              leaves_at_depth_d[d - 1]))]
    leaves_kept = leaves_sorted[int(np.ceil(prune_rate * len(cost_at_leaf))):]
    leaves_removed = leaves_sorted[:int(np.ceil(prune_rate * len(cost_at_leaf)))]
    G.remove_nodes_from(leaves_removed)
    leaves_at_depth_d[d - 1] = leaves_kept


def tree_grow_root(G, leaves_at_depth_d, layers):
    for architecture in layers:
        G.add_edge('ROOT', architecture)
        leaves_at_depth_d[1].append(architecture)


def tree_grow(G, leaves_at_depth_d, d, layers):
    for architecture in leaves_at_depth_d[d - 1]:
        for new_layer in layers:
            new_architecture = ':'.join([architecture, new_layer])
            comm_check = new_architecture.split(':')
            if comm_check[-2] != comm_check[-1]:
                G.add_node(new_architecture)
                G.add_edge(architecture, new_architecture)
                leaves_at_depth_d[d].append(new_architecture)


def tree_cost_of_path(G, leaf):
    paths = nx.shortest_path(G, 'ROOT', leaf)
    return sum([G.nodes[node]['W'] for node in paths])


def construct_circuit_from_leaf(leaf, nqubits, nclasses, dev):
    architecture = leaf.split(':')
    # TODO: GENERALIZE FOR ARBITRARY NUMBER OF CLASSES

    embedding_circuit = architecture.pop(0)
    # print(embedding_circuit)
    def circuit_from_architecture(params, features):
        string_to_embedding_mapping[embedding_circuit](features, dev.wires)
        for d, component in enumerate(architecture):
            string_to_layer_mapping[component](list(range(nqubits)), params[:, d])
        return [qml.expval(qml.PauliZ(nc)) for nc in range(nclasses)]

    params_shape = (nqubits, len(architecture))

    return qml.QNode(circuit_from_architecture, dev), params_shape


def run_tree_architecture_search(config):
    NQUBITS = config['nqubits']
    NCLASSES = config['nclasses']
    NSAMPLES = config['n_samples']
    dev = qml.device("default.qubit", wires=NQUBITS)
    MIN_TREE_DEPTH = config['min_tree_depth']
    MAX_TREE_DEPTH = config['max_tree_depth']

    PRUNE_DEPTH_STEP = config['prune_step']  # EVERY ith step is a prune step
    PRUNE_RATE = config['prune_rate']  # Percentage of nodes to throw away at each layer
    PLOT_INTERMEDIATE_TREES = config['plot_trees']

    assert MIN_TREE_DEPTH < MAX_TREE_DEPTH, 'MIN_TREE_DEPTH must be smaller than MAX_TREE_DEPTH'
    assert NQUBITS >= NCLASSES, 'The number of qubits must be equal or larger than the number of classes'
    # TODO: ADD DATA LOADER HERE
    if config['data_set'] == 'circles':
        X_train, y_train = datasets.make_circles(n_samples=NSAMPLES, factor=.5, noise=.05)
    elif config['data_set'] == 'moons':
        X_train, y_train = datasets.make_moons(n_samples=NSAMPLES, noise=.05)
    #convert to -1 1
    X_train = np.multiply(1.0, np.subtract(np.multiply(np.divide(np.subtract(X_train, X_train.min()),
                                                                 (X_train.max() - X_train.min())), 2.0), 1.0))
    # one hot encode
    y_train_ohe = np.zeros((y_train.size, y_train.max() + 1))

    y_train_ohe[np.arange(y_train.size), y_train] = 1
    # print(noisy_data)
    G = nx.DiGraph()

    G.add_node("ROOT")
    nx.set_node_attributes(G, {'ROOT': 0.0}, 'W')
    # TODO: ARE THESE LAYERS ENOUGH?
    possible_layers = ['ZZ', 'X', 'Y']
    possible_embeddings = ['E1', ]
    leaves_at_depth_d = dict(zip(range(MAX_TREE_DEPTH), [[] for _ in range(MAX_TREE_DEPTH)]))
    leaves_at_depth_d[0].append('ROOT')
    # Iteratively construct tree, pruning at set rate
    for d in range(1, MAX_TREE_DEPTH):
        print(f"Depth = {d}")
        if PLOT_INTERMEDIATE_TREES:
            fig, axs = plt.subplots(1, 1)
            fig.set_size_inches(16, 8)
            pos = graphviz_layout(G, prog='dot')
            colors = nx.get_node_attributes(G, 'W')
            node_color = list(colors.values())
            vmin = min(node_color)
            vmax = max(node_color)
            nx.draw(G, pos=pos, arrows=True, with_labels=True, cmap='OrRd', node_color=node_color, linewidths=1,
                    vmin=vmin, vmax=vmax, ax=axs)
            axs.collections[0].set_edgecolor("#000000")

            sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))

            cb = plt.colorbar(sm)
            cb.set_label('W-cost')
            axs.set_title('Tree of costs')
            plt.show()
        # TODO: ADD EMBEDDINGS AT THE ROOT, PARSE THIS PROPERLY IN construct_circuit_from_leaf()
        if d < MIN_TREE_DEPTH:
            if d == 1:
                tree_grow_root(G, leaves_at_depth_d, possible_embeddings)
            else:
                tree_grow(G, leaves_at_depth_d, d, possible_layers)
            # At the embedding level we don't need to train because there are no params.
            for v in leaves_at_depth_d[d]:
                nx.set_node_attributes(G, {v: 1.0}, 'W')
                # if d == 1:
                #     # RUN CIRCUITS HERE
                #     circuit, pshape = construct_circuit_from_leaf(v, NQUBITS, NCLASSES, dev)
                #     print(pshape)
        else:
            if not (d - MIN_TREE_DEPTH) % PRUNE_DEPTH_STEP:
                print('Prune Tree')
                tree_prune(G, leaves_at_depth_d, d, PRUNE_RATE)
                print('Grow Pruned Tree')
                tree_grow(G, leaves_at_depth_d, d, possible_layers)
                for v in leaves_at_depth_d[d]:

                    # RUN CIRCUITS HERE
                    circuit, pshape = construct_circuit_from_leaf(v, NQUBITS, NCLASSES, dev)
                    # TODO: RUN TRAINING
                    # TODO: CALCULATE NUMBER OF CNOTS
                    # TODO: ADD W-COST AS ATTRIBUTE
                    print(f'Training leaf {v}')
                    w_cost = train_circuit(circuit, pshape, X_train, y_train_ohe, 'accuracy', **config)
                    nx.set_node_attributes(G, {v: w_cost}, 'W')

            else:
                print('Grow Tree')
                tree_grow(G, leaves_at_depth_d, d, possible_layers)
                for v in leaves_at_depth_d[d]:
                    nx.set_node_attributes(G, {v: arbitrary_cost_fn()[0]}, 'W')
                    # RUN CIRCUITS HERE
                    circuit, pshape = construct_circuit_from_leaf(v, NQUBITS, NCLASSES, dev)
                    # TODO: RUN TRAINING
                    # TODO: CALCULATE NUMBER OF CNOTS
                    # TODO: ADD W-COST AS ATTRIBUTE
                    print(f'Training leaf {v}')
                    w_cost = train_circuit(circuit, pshape, X_train, y_train_ohe, 'accuracy', **config)
                    nx.set_node_attributes(G, {v: w_cost}, 'W')
