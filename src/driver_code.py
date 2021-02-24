from subarchitecture_tree_search import run_tree_architecture_search
import pennylane as qml

import os
import pickle

if __name__ == "__main__":
    EXPERIMENT_NAME = 'alpha'
    if not os.path.exists('data'):
        os.mkdir('data/')
    data_path = f'data/{EXPERIMENT_NAME}'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    config = {'nqubits': 3,
              'nclasses': 2,
              'min_tree_depth': 3,
              'max_tree_depth': 8,
              'prune_rate': 0.3,
              'prune_step': 3,
              'plot_trees': False,
              'data_set': 'moons',
              'nsteps': 20,
              'opt': qml.AdamOptimizer,
              'batch_size': 25,
              'n_samples': 1500,
              'learning_rate': 0.01,
              'save_frequency': 1,
              'save_path': data_path
              }
    with open(data_path + '/config.pickle', 'wb') as f:
        pickle.dump(config, f)
    run_tree_architecture_search(config)
