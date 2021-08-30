# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
================================================
Aer Jobs (:mod:`qiskit.providers.aer.jobs`)
================================================

.. currentmodule:: qiskit.providers.aer.jobs

This module contains classes and functions to manage Aer jobs.

Running with Threadpool and DASK
================================

Qiskit Aer runs simulation jobs on a single-worker Python multiprocessing ThreadPool executor
so that all parallelization is handled by low-level OpenMP and CUDA code.
However to customize job-level parallel execution of multiple circuits a user can specif
a custom multiprocessing executor and control the splitting of circuits using
the ``executor`` and ``max_job_size`` backend options.
For large scale job parallelization on HPC clusters Qiskit Aer executors support
the distributed Clients from the `DASK <http://dask.org>`_.


Installation of DASK packages with Aer
---------------------------------------
If you want to install dask client at the same time as Qiskit Aer,
please add `dask` option as follows.
This option installs Aer, dask, and  distributed packages.

.. code-block:: sh

    pip install .[dask]

Usage of executor
-----------------
To use Threadpool or DASK as an executor, you need to set
``executor`` and ``max_job_size`` by ``set_options`` function.
If both ``executor`` (default None) and `max_job_size` (default None) are set,
Aer splits the multiple circuits to some chunk of circuits and submits them to the executor.
``max_job_size`` can control the number of splitting circuits.
When ``max_job_size`` is set to 1, multiple circuits are split into
one circuit and distributed to the executor.
If user executes 60 circuits with the executor and `max_job_size=1`,
Aer splits it to 1 circuit x 60 jobs.
If 60 circuits and `max_job_size=2`, Aer splits it to 2 circuits x 30 jobs.

**Example: Threadpool execution**

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
    circ.u1(pi/2,2)
    circ.measure([0, 1, 2], [0, 1 ,2])

    circ2 = qiskit.QuantumCircuit(15, 15)
    circ2.h(0)
    circ2.cx(0, 1)
    circ2.cx(1, 2)
    circ2.u1(pi/2,2)
    circ2.measure([0, 1, 2], [0, 1 ,2])
    circ_list = [circ, circ2]

    qbackend = AerSimulator()
    #Set executor and max_job_size
    exc = ThreadPoolExecutor(max_workers=2)
    qbackend.set_options(executor=exc)
    qbackend.set_options(max_job_size=1)

    result = qbackend.run(circ_list).result()

**Example: Dask execution**

Dask client creates multi-processes so you need to
guard it by ``if __name__ == "__main__":`` block.

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
        circ.u1(pi/2,2)
        circ.measure([0, 1, 2], [0, 1 ,2])

        circ2 = qiskit.QuantumCircuit(15, 15)
        circ2.h(0)
        circ2.cx(0, 1)
        circ2.cx(1, 2)
        circ2.u1(pi/2,2)
        circ2.measure([0, 1, 2], [0, 1 ,2])

        circ_list = [circ, circ2]

        exc = Client(address=LocalCluster(n_workers=2, processes=True))
        #Set executor and max_job_size
        qbackend = AerSimulator()
        qbackend.set_options(executor=exc)
        qbackend.set_options(max_job_size=1)

        result = qbackend.run(circ_list).result()

    if __name__ == '__main__':
        q_exec()

Classes
=======

The following are the classes used to management job submitting.

.. autosummary::
    :toctree: ../stubs/

    AerJob
    AerJobSet

"""

from .aerjob import AerJob
from .aerjobset import AerJobSet
from .utils import split_qobj
