import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import itertools as it
import matplotlib.pyplot as plt


def arbitrary_cost_fn():
    return np.random.rand(1)


def non_commuting_check(architecture):
    for g1, g2 in zip(architecture[:-1], architecture[1:]):
        if g1 == g2:
            return False
    return True


depth = 8

MIN_TREE_DEPTH = 3
PRUNE_DEPTH_STEP = 3  # EVERY ith step is a prune step
PRUNE_RATE = 0.5 # Percentage of nodes to throw away at each layer
MAX_TREE_DEPTH = depth
assert MIN_TREE_DEPTH < MAX_TREE_DEPTH, 'MIN_TREE_DEPTH must be smaller than MAX_TREE_DEPTH'

G = nx.DiGraph()

G.add_node("ROOT")
nx.set_node_attributes(G, {'ROOT': 0.0}, 'W')

num_2_string_dict = {0: 'ZZ', 1: 'X', 2: 'Y'}
n_architecures = 0
leafs_at_depth_d = dict(zip(range(MAX_TREE_DEPTH), [[] for _ in range(MAX_TREE_DEPTH)]))
leafs_at_depth_d[0].append('ROOT')
## Iteratively construct tree, pruning at set rate

for d in range(1, MAX_TREE_DEPTH):
    print(f"Depth = {d}")
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(16, 8)
    pos = graphviz_layout(G, prog='dot')
    node_size = []
    colors = nx.get_node_attributes(G, 'W')
    node_color = list(colors.values())
    vmin = min(node_color)
    vmax = max(node_color)
    nx.draw(G, pos=pos, arrows=True, with_labels=False, cmap='OrRd', node_color=node_color, linewidths=1,
            vmin=vmin, vmax=vmax, ax=axs)
    axs.collections[0].set_edgecolor("#000000")

    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))

    cb = plt.colorbar(sm)
    cb.set_label('W-cost')
    axs.set_title('Tree of costs')
    plt.show()
    if d < MIN_TREE_DEPTH:
        for architecture in it.product(list(range(len(num_2_string_dict.values()))), repeat=d):
            if non_commuting_check(architecture):
                n_architecures += 1
                name_d = ':'.join([num_2_string_dict[s] for s in architecture])
                if d == 1:
                    G.add_edge('ROOT', name_d)
                else:
                    name_d = ':'.join([num_2_string_dict[s] for s in architecture])
                    G.add_node(name_d)
                    name_d_min_1 = ':'.join([num_2_string_dict[s] for s in architecture[:-1]])
                    G.add_edge(name_d_min_1, name_d)
                leafs_at_depth_d[d].append(name_d)
        for v in leafs_at_depth_d[d]:
            nx.set_node_attributes(G, {v: arbitrary_cost_fn()[0]}, 'W')
            # RUN CIRCUITS HERE
    else:
        if not (d - MIN_TREE_DEPTH) % PRUNE_DEPTH_STEP:
            cost_at_leaf = []
            for leaf in leafs_at_depth_d[d - 1]:
                paths = nx.shortest_path(G, 'ROOT', leaf)
                total_cost = sum([G.nodes[node]['W'] for node in paths])
                cost_at_leaf.append(total_cost)
            leaves_sorted = [x for _, x in sorted(zip(cost_at_leaf,
                                                      leafs_at_depth_d[d - 1]))]
            leaves_kept = leaves_sorted[:int(np.ceil(PRUNE_RATE * len(cost_at_leaf)))]
            leaves_removed = leaves_sorted[int(np.ceil(PRUNE_RATE * len(cost_at_leaf))):]
            G.remove_nodes_from(leaves_removed)
            leafs_at_depth_d[d - 1] = leaves_kept
            print('Grow Pruned Tree')
            for architecture in leafs_at_depth_d[d - 1]:
                for new_layer in ['ZZ', 'X', 'Y']:
                    new_architecture = ':'.join([architecture, new_layer])
                    comm_check = new_architecture.split(':')
                    if comm_check[-2] != comm_check[-1]:
                        n_architecures += 1
                        G.add_node(new_architecture)
                        G.add_edge(architecture, new_architecture)
                        leafs_at_depth_d[d].append(new_architecture)
            for v in leafs_at_depth_d[d]:
                nx.set_node_attributes(G, {v: arbitrary_cost_fn()[0]}, 'W')
                # RUN CIRCUITS HERE
        else:
            print('Grow Tree')
            for architecture in leafs_at_depth_d[d - 1]:
                for new_layer in ['ZZ', 'X', 'Y']:
                    new_architecture = ':'.join([architecture, new_layer])
                    comm_check = new_architecture.split(':')
                    if comm_check[-2] != comm_check[-1]:
                        n_architecures += 1
                        G.add_node(new_architecture)
                        G.add_edge(architecture, new_architecture)
                        leafs_at_depth_d[d].append(new_architecture)
            for v in leafs_at_depth_d[d]:
                nx.set_node_attributes(G, {v: arbitrary_cost_fn()[0]}, 'W')
                # RUN CIRCUITS HERE