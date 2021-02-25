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
              'min_tree_depth': 4,
              'max_tree_depth': 5,
              'prune_rate': 0.15,
              'prune_step': 3,
              'plot_trees': False,
              'data_set': 'moons',
              'nsteps': 5,
              'optim': qml.AdamOptimizer,
              'batch_sizes': [8,16,32,64],
              'n_samples': 1500,
              'learning_rates': [0.001,0.005,0.01],
              'save_frequency': 1,
              'save_path': data_path,
              'save_timing': True,
              'circuit_type':'schuld',
              'Tmax': [100,100,100],
              'inf_time':'numcnots',
              'fill':'redundant', # or 'pad'
              'rate_type': 'accuracy', # or 'batch_cost'
              'readout_layer': 'one_hot',  #or 'weighted_neuron'
              }

    # Save the configuration file so that we can remember what we did
    with open(data_path + '/config.pickle', 'wb') as f:
        pickle.dump(config, f)

    # Execute the algorithm
    run_tree_architecture_search(config)
