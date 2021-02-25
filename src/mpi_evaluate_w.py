from mpi4py import MPI
import numpy
from train_utils import evaluate_w
from circuit_utils import construct_circuit_from_leaf
import pickle

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

architectures = None
save_path = None
data = comm.scatter(architectures, root=0)
leaf = data[0]
save_path = data[1]

with open(save_path, 'rb') as pdata:
    pickled_data_for_MPI = pickle.load(pdata)
NQUBITS, NCLASSES, dev, config, X_train, y_train_ohe = pickled_data_for_MPI

circuit, pshape, numcnots = construct_circuit_from_leaf(leaf, NQUBITS, NCLASSES, dev, config)
config['numcnots'] = numcnots
w_cost, _ = evaluate_w(circuit, pshape, X_train, y_train_ohe, **config)
# print(w_cost)
# print(type(w_cost))
comm.gather(w_cost, root=0)

comm.Disconnect()