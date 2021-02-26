from qose.subarchitecture_tree_search import run_tree_architecture_search
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
    # Create a configuration file for the tree prune algorithm
    config = {'nqubits': 2,
              'min_tree_depth': 3,
              'max_tree_depth': 4,
              'prune_rate': 0.3,
              'prune_step': 2,
              'plot_trees': False,
              'data_set': 'moons',
              'nsteps': 8,
              'optim': qml.AdamOptimizer,
              'batch_sizes': [8,16,32],
              'n_samples': 1000,
              'learning_rates': [0.1,0.2],
              'save_frequency': 1,
              'save_path': data_path,
              'save_timing': False,
              'circuit_type':'schuld',
              'Tmax': [100,100,100],
              'inf_time':'timeit',
              'fill':'redundant', # or 'pad'
              'rate_type': 'batch_cost', # 'accuracy' or 'batch_cost'
              'readout_layer': 'one_hot',  #'one_hot' or 'weighted_neuron'
              }

    # Save the configuration file so that we can remember what we did
    with open(data_path + '/config.pickle', 'wb') as f:
        pickle.dump(config, f)

    # Execute the algorithm
    run_tree_architecture_search(config)
