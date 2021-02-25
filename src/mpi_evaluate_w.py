from mpi4py import MPI
import numpy

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

architectures = None
leaf = comm.scatter(architectures, root=0)
print(leaf)
print(f'RANK {rank}: I have received {leaf}')

PI_sent = rank * 1
# PI_received = numpy.empty(5, 'd')
# comm.Reduce([PI, MPI.DOUBLE], None,
#             op=MPI.SUM, root=0)
# comm.Gather([PI, MPI.DOUBLE], root=0)
comm.gather(PI_sent, root=0)

comm.Disconnect()