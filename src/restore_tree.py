import pickle
from plot_utils import plot_tree

import networkx as nx
import os

if __name__ == "__main__":
    EXPERIMENT_NAME = 'alpha'
    datapath = f'data/{EXPERIMENT_NAME}/'
    with open(datapath + 'config.pickle', 'rb') as f:
        config = pickle.load(f)

    restore_depth = 6

    with open(datapath + f'tree_depth_{restore_depth}.pickle', 'rb') as f:
        G = pickle.load(f)
    plot_tree(G, labels=False)