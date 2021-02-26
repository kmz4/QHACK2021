### What is our code doing right now?

What follows is an explanation of the code used for QOSE.

The main file to run everything is `driver_code.py`. This script contains a configuration dictionary that gets send to 
the function `run_tree_architecture_search`. In `driver_code.py`, we automatically create a folder for storing 
the tree data and the configuration of the experiment. This folder will have the name that you assign to
`EXPERIMENT_NAME` (line 8). This setup will hopefully to run systematic experiments by keeping track of all the data.

In order to organize the code, there are now three files, `circuit_utils.py`, `train_utils.py` and `plot_utils.py`, that
contain the all the functions we need for constructing circuits, handling training and plotting data respectively. 

The tree pruning algorithm with the function `run_tree_architecture_search` can be found in 
`subarchitecture_tree_search.py`. The tree that is used to represent all the circuits is created with the Python library
 `networkx`. Everytime we add a layer to the circuit, we save the `networkx` tree of respective depth. Hence, 
 running the driver code to depth 3 with 
 ```python
EXPERIMENT_NAME = 'alpha'
```
will result in the folder structure

    ├── src
        ├── data
                ├── alpha
                        ├── config.pickle
                        ├── tree_depth_1.pickle
                        ├── tree_depth_2.pickle
                        ├── tree_depth_3.pickle
        ├── circuit_utils.py
        ├── driver_code.py
        ├── ...
This data can be retrieved in the script `restore_tree.py`. And will result in a plot of the circuit Tree from that
experiment.
  
## Paralellization on AWS

In an MPI enabled environment, we can call `driver_code_par.py` instead of `driver_code.py` with the following 
command:
```bash
mpiexec -n 1 --oversubscribe python3 driver_code_par.py
```
Which will parallelize the leaf calculation in the tree. For our project, this was confirmed to work on a
ml.m5.24xlarge AWS instance, that allowed for parallel calculation of up to 90 circuits, constructing a massive tree of 
possible architectures as a result: