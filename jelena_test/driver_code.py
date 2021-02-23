from subarchitecture_tree_search import run_tree_architecture_search


if __name__=="__main__":

    config = {'nqubits': 3,
              'nclasses': 2,
              'loss_function': 'hinge',
              'min_tree_depth':3,
              'max_tree_depth': 8,
              'prune_rate': 0.3,
              'prune_step': 3,
              'plot_trees': False,
              'data_set': 'moons',
              'nsteps': 20,
              'opt': qml.AdamOptimizer,
              'opt_opts': {'stepsize':0.01},
              'batch_size':100,
              'n_samples':1500
              }
    run_tree_architecture_search(config)