from mpi4py import MPI
import numpy
import sys

comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['cpi.py'],
                           maxprocs=5)

N = numpy.array([1,2,3,4,5], 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
PI_sent = numpy.empty(1, 'd')
PI_received = numpy.empty(5, 'd')
# comm.Reduce(None, [PI, MPI.DOUBLE],
#             op=MPI.SUM, root=MPI.ROOT)
comm.Gather(PI_sent, PI_received, root=MPI.ROOT)
print(PI_received)
# print(PI)

comm.Disconnect()