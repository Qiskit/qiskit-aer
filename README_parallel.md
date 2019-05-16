# qiskit-aer-parallel


To enable GPUs, 
cmake AER_PARALLEL=true AER_CUDA=true

To enable MPI,
cmake AER_PARALLEL=true AER_MPI=true

To enable MPI and GPUs,
cmake AER_PARALLEL=true AER_MPI=true AER_CUDA=true




To run with MPI, please set following environmental variable to tell number of processes per node.

export QSIM_PROC_PER_NODE=2




