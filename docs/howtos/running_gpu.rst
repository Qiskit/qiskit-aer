.. _running_gpu:

Running with multiple-GPUs and/or multiple nodes
================================================

Qiskit Aer parallelizes simulations by distributing quantum states into
distributed memory space. To decrease data transfer between spaces the
distributed states are managed as chunks that is a sub-state for smaller
qubits than the input circuits.

For example, 30-qubits circuit is distributed into 2^10 chunks with
20-qubits.

To decrease data exchange between chunks and also to simplify the
implementation, we are applying cache blocking technique. This technique
allows applying quantum gates to each chunk independently without data
exchange, and serial simulation codes can be reused without special
implementation. Before the actual simulation, we apply transpilation to
remap the input circuits to the equivalent circuits that has all the
quantum gates on the lower qubits than the chunk’s number of qubits. And
the (noiseless) swap gates are inserted to exchange data.

Please refer to this paper (https://arxiv.org/abs/2102.02957) for more
detailed algorithm and implementation of parallel simulation.

So to simulate by using multiple GPUs or multiple nodes on the cluster,
following configurations should be set to backend options. (If there is
not enough memory to simulate the input circuit, Qiskit Aer
automatically set following options, but it is recommended to explicitly
set them)

-  blocking_enable

should be set to True for distributed parallelization. (Default = False)

-  blocking_qubits

this flag sets the qubit number for chunk, should be smaller than the
smallest memory space on the system (i.e. GPU). Set this parameter to
satisfy
``sizeof(complex)*2^(blocking_qubits+4) < size of the smallest memory space``
in byte.

Here is an example how we parallelize simulation with multiple GPUs.

.. code:: python

   sim = AerSimulator(method='statevector', device='GPU')
   circ = transpile(QuantumVolume(qubit, 10, seed = 0))
   circ.measure_all()
   result = execute(circ, sim, shots=100, blocking_enable=True, blocking_qubits=23).result()

To run Qiskit Aer with Python script with MPI parallelization, MPI
executer such as mpirun should be used to submit a job on the cluster.
Following example shows how to run Python script using 4 processes by
using mpirun.

.. code:: sh

    mpirun -np 4 python example.py

MPI_Init function is called inside Qiskit Aer, so you do not have to
manage MPI processes in Python script. Following metadatas are useful to
find on which process is this script running.

-  num_mpi_processes : shows number of processes using for this
   simulation
-  mpi_rank : shows zero based rank (process ID)

Here is an example how to get my rank.

.. code:: python

   sim = AerSimulator(method='statevector', device='GPU')
   result = execute(circuit, sim, blocking_enable=True, blocking_qubits=23).result()
   dict = result.to_dict()
   meta = dict['metadata']
   myrank = meta['mpi_rank']

Multiple shots are also distributed to multiple nodes when setting
``device=GPU`` and ``batched_shots_gpu=True``. The results are
distributed to each processes.

Note : In the script, make sure that the same random seed should be used
for all processes so that the consistent circuits and parameters are
passed to Qiskit Aer. To do so add following option to the script.

.. code:: python

   from qiskit.utils import algorithm_globals
   algorithm_globals.random_seed = consistent_seed_to_all_processes


GPU simulation precision and performance considerations
-------------------------------------------------------

Running quantum circuit simulations on a GPU does not automatically guarantee a
speedup compared with CPU execution. Actual performance depends on multiple factors,
including the GPU model and architecture, CPU baseline capabilities, circuit size
(number of qubits and gate depth), simulation method, and workload. For smaller
circuits or shallow depths, GPU kernel launch overhead and host-to-device data transfers
may offset any computational advantage.

A key factor affecting GPU simulation performance is floating-point precision:

- **Default precision**: By default, Qiskit Aer uses double precision
  (``precision="double"``) where applicable (such as in ``statevector``,
  ``density_matrix``, and ``unitary`` simulation methods). In double precision,
  each complex amplitude is represented using two 64-bit floating-point numbers
  (16 bytes total).

- **Hardware FP64 vs. FP32 performance**: Different GPU architectures have
  substantially different performance characteristics for double-precision (FP64)
  versus single-precision (FP32) arithmetic. Even on high-performance computing
  accelerators such as the NVIDIA V100, which feature dedicated FP64 hardware,
  single-precision compute throughput is higher than double precision, and memory
  traffic is halved. On many other GPUs (such as consumer or workstation GPUs),
  FP64 throughput is often significantly more constrained relative to FP32.
  Consequently, running double-precision simulations on such hardware can result
  in slower execution than anticipated.

- **Configuring single precision**: Users performing GPU statevector simulations
  should consider setting ``precision="single"`` when appropriate. Single precision
  halves the memory requirement to 8 bytes per complex amplitude, allowing larger
  quantum states to fit in GPU memory and potentially providing substantial performance
  improvements on GPUs where single-precision execution is faster.

- **Numerical accuracy trade-offs**: While single precision can improve GPU
  performance and reduce memory usage, it provides lower numerical accuracy
  (approximately 7 decimal digits of precision compared to roughly 15–17 digits
  for double precision). Single precision is not universally recommended, and users
  should not use single precision in scenarios such as:

  - Deep circuits with many gate layers, where small floating-point roundoff errors
    can accumulate.
  - Algorithms highly sensitive to delicate phase cancellations, small probability
    amplitudes, or tight error tolerances (such as quantum phase estimation or precision
    quantum chemistry simulations).
  - Validation workloads where strict numerical fidelity to the theoretical statevector
    is required.

Single precision does not guarantee a speedup in every scenario, and users should
benchmark their specific circuits and workloads on the target hardware to determine
the optimal balance between performance and numerical accuracy.

Example: Setting precision on AerSimulator
''''''''''''''''''''''''''''''''''''''''''

To configure an Aer GPU simulator with single precision:

.. code:: python

    from qiskit_aer import AerSimulator

    # Initialize GPU statevector simulator with single precision
    simulator = AerSimulator(method="statevector", device="GPU", precision="single")

Alternatively, precision can be updated on an existing simulator instance:

.. code:: python

    simulator.set_options(precision="single")
