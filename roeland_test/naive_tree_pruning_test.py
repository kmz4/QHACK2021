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


depth = 6

MIN_TREE_DEPTH = 2
MAX_TREE_DEPTH = 6
G = nx.DiGraph()

G.add_node("ROOT")

num_2_string_dict = {0: 'ZZ', 1: 'X', 2: 'Y'}
n_architecures = 0
leafs_at_depth_d = dict(zip(range(MAX_TREE_DEPTH), [[] for _ in range(MAX_TREE_DEPTH)]))
for architecture in it.product(list(range(3)), repeat=depth):
    if non_commuting_check(architecture):
        n_architecures += 1
        # print(architecture)
        for d, layer in enumerate(architecture):
            name_d = ':'.join([num_2_string_dict[s] for s in architecture[:d + 1]])
            # print(name_d)
            name_d_min_1 = ':'.join([num_2_string_dict[s] for s in architecture[:d]])
            G.add_node(name_d)
            if d > 0:
                G.add_edge(name_d_min_1, name_d)
            else:
                G.add_edge('ROOT', name_d)
            leafs_at_depth_d[d].append(name_d)
cmap = plt.get_cmap('Blues')
nx.set_node_attributes(G, {'ROOT': 0.0}, 'W')
for d in range(depth):
    for v in leafs_at_depth_d[d]:
        nx.set_node_attributes(G, {v: arbitrary_cost_fn()[0]}, 'W')

for leaf in leafs_at_depth_d[3]:
    paths = nx.shortest_path(G, 'ROOT', leaf)
    total_cost = sum([G.nodes[node]['W'] for node in paths])
    print(total_cost)
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

# sm._A = []
cb = plt.colorbar(sm)
cb.set_label('W-cost')
axs.set_title('Tree of costs')
plt.show()
