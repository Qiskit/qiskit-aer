.. _dask:

Running with Threadpool and DASK
================================

Qiskit Aer runs simulation jobs on a single-worker Python multiprocessing ThreadPool executor
so that all parallelization is handled by low-level OpenMP and CUDA code.
However to customize job-level parallel execution of multiple circuits a user can specify
a custom multiprocessing executor and control the splitting of circuits using
the ``executor`` and ``max_job_size`` backend options.
For large scale job parallelization on HPC clusters Qiskit Aer executors support
the distributed Clients from the `DASK <http://dask.org>`__.

Installation of DASK packages with Aer
---------------------------------------

If you want to install dask client at the same time as Qiskit Aer,
please add the ``dask`` extra as follows.
This option installs Aer, dask, and  distributed packages.

.. code-block:: sh

    pip install .[dask]

Usage of executor
-----------------

To use Threadpool or DASK as an executor, you need to set
``executor`` and ``max_job_size`` by ``set_options`` function.
If both ``executor`` (default None) and ``max_job_size`` (default None) are set,
Aer splits the multiple circuits to some chunk of circuits and submits them to the executor.
``max_job_size`` can control the number of splitting circuits.
When ``max_job_size`` is set to 1, multiple circuits are split into
one circuit and distributed to the executor.
If a user executes 60 circuits with the executor and ``max_job_size=1``,
Aer splits it as 60 jobs each of 1 circuit.
If there are 60 circuits and ``max_job_size=2``, Aer splits it as 30 jobs, each with 2 circuits.

Example: Threadpool execution
'''''''''''''''''''''''''''''

.. code-block:: python

    import qiskit
    from concurrent.futures import ThreadPoolExecutor
    from qiskit.providers.aer import AerSimulator
    from math import pi

    # Generate circuit
    circ = qiskit.QuantumCircuit(15, 15)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.p(pi/2, 2)
    circ.measure([0, 1, 2], [0, 1 ,2])

    circ2 = qiskit.QuantumCircuit(15, 15)
    circ2.h(0)
    circ2.cx(0, 1)
    circ2.cx(1, 2)
    circ2.p(pi/2, 2)
    circ2.measure([0, 1, 2], [0, 1 ,2])
    circ_list = [circ, circ2]

    qbackend = AerSimulator()
    # Set executor and max_job_size
    exc = ThreadPoolExecutor(max_workers=2)
    qbackend.set_options(executor=exc)
    qbackend.set_options(max_job_size=1)
    result = qbackend.run(circ_list).result()

Example: Dask execution
'''''''''''''''''''''''

The Dask client uses ``multiprocessing`` so you need to
guard it by an ``if __name__ == "__main__":`` block.

.. code-block:: python

    import qiskit
    from qiskit.providers.aer import AerSimulator
    from dask.distributed import LocalCluster, Client
    from math import pi
    def q_exec():
        # Generate circuits
        circ = qiskit.QuantumCircuit(15, 15)
        circ.h(0)
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.p(pi/2, 2)
        circ.measure([0, 1, 2], [0, 1 ,2])

        circ2 = qiskit.QuantumCircuit(15, 15)
        circ2.h(0)
        circ2.cx(0, 1)
        circ2.cx(1, 2)
        circ2.p(pi/2, 2)
        circ2.measure([0, 1, 2], [0, 1 ,2])

        circ_list = [circ, circ2]

        exc = Client(address=LocalCluster(n_workers=2, processes=True))
        # Set executor and max_job_size
        qbackend = AerSimulator()
        qbackend.set_options(executor=exc)
        qbackend.set_options(max_job_size=1)
        result = qbackend.run(circ_list).result()


    if __name__ == '__main__':
        q_exec()

