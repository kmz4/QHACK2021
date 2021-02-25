from subarchitecture_tree_search import run_tree_architecture_search
import pennylane as qml

import os
import pickle

if __name__ == "__main__":
    # Create a unique name for your experiment
    EXPERIMENT_NAME = 'alpha'

    # Create a directory to store the data
    if not os.path.exists('data'):
        os.mkdir('data/')

    data_path = f'data/{EXPERIMENT_NAME}'
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Create a configuration file for the tree prune algorithm
    config = {'nqubits': 3,
              'min_tree_depth': 3,
              'max_tree_depth': 8,
              'prune_rate': 0.3,
              'prune_step': 3,
              'plot_trees': False,
              'data_set': 'moons',
              'nsteps': 5,
              'opt': qml.AdamOptimizer,
              'batch_sizes': [25],
              'n_samples': 1500,
              'learning_rates': [0.01],
              'save_frequency': 1,
              'save_path': data_path,
              'Tmax': [100,100,100],
              'rate_type': 'accuracy',
              'inf_time':'timeit'
              }

    # Save the configuration file so that we can remember what we did
    with open(data_path + '/config.pickle', 'wb') as f:
        pickle.dump(config, f)

    # Execute the algorithm
    run_tree_architecture_search(config)
