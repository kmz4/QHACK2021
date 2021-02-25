from mpi4py import MPI
import numpy

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

N = numpy.empty(5, dtype='i')
comm.Bcast([N, MPI.INT], root=0)
print(f'RANK {rank}: I have received {N} and will return {N[rank]}')

PI_sent = numpy.empty(1, 'd')
PI_sent[0] = N[rank]
PI_received = numpy.empty(5, 'd')
# comm.Reduce([PI, MPI.DOUBLE], None,
#             op=MPI.SUM, root=0)
# comm.Gather([PI, MPI.DOUBLE], root=0)
comm.Gather(PI_sent, PI_received, root=0)

comm.Disconnect()