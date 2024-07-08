:orphan:

###############
Getting started
###############

Installation
============
Qiskit Aer depends on the main Qiskit package which has its own
`Qiskit Installation guide <https://docs.quantum.ibm.com/start/install>`__ detailing the
installation options for Qiskit and its supported environments/platforms. You should refer to
that first. Then the information here can be followed which focuses on the additional installation
specific to Qiskit Aer.


.. tab-set::

    .. tab-item:: Start locally

      The simplest way to get started is to follow the installation guide for Qiskit `here <https://docs.quantum.ibm.com/start/install>`__

      In your virtual environment where you installed Qiskit, add ``qiskit-aer``, e.g.:

      .. code:: sh

         pip install qiskit-aer

      **Installing GPU support**

      In order to install and run the GPU supported simulators on Linux, you need CUDA® 10.1 or newer 
      previously installed. CUDA® itself would require a set of specific GPU drivers. 
      Please follow CUDA® installation procedure in the NVIDIA® `web <https://www.nvidia.com/drivers>`_.  

      If you want to install our GPU supported simulators, you have to install this other package:

      .. code:: sh

         pip install qiskit-aer-gpu

      This will overwrite your current qiskit-aer package installation giving you the same functionality found 
      in the canonical qiskit-aer package, plus the ability to run the GPU supported 
      simulators: statevector, density matrix, and unitary.

      *Note: This package is only available on x86_64 Linux. 
      For other platforms that have CUDA support you will have to build from source.* 


    .. tab-item:: Install from source

      
      Installing Qiskit Aer from source allows you to access the most recently
      updated version under development instead of using the version in the Python Package
      Index (PyPI) repository. This will give you the ability to inspect and extend
      the latest version of the Qiskit Aer code more efficiently.

      Since Qiskit Aer depends on Qiskit, and its latest changes may require new or changed
      features of Qiskit, you should first follow Qiskit's `"Install from source"` instructions `here <https://docs.quantum.ibm.com/start/install-qiskit-source>`__

      .. raw:: html

         <h2>Installing Qiskit Aer from Source</h2>
      

      Clone the ``Qiskit Aer`` repo via *git*.

      .. code:: sh

         git clone https://github.com/Qiskit/qiskit-aer

      The common dependencies can then be installed via *pip*, using the ``requirements-dev.txt`` file, e.g.:

      .. code:: sh

         cd qiskit-aer
         pip install -r requirements-dev.txt

      As any other Python package, we can install from source code by just running:

      .. code:: sh

         qiskit-aer$ pip install .

      This will build and install ``Aer`` with the default options which is
      probably suitable for most of the users. There’s another Pythonic
      approach to build and install software: build the wheels distributable
      file.

      .. code:: sh

         qiskit-aer$ pip install build
         qiskit-aer$ python -I -m build --wheel


      See `here <https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md#install-from-source>`__ 
      for detailed installation information.

      .. raw:: html

         <h2>Building with GPU support</h2>

      Qiskit Aer can exploit GPU’s horsepower to accelerate some simulations,
      specially the larger ones. GPU access is supported via CUDA® (NVIDIA®
      chipset), so to build with GPU support, you need to have CUDA® >= 10.1
      preinstalled. See install instructions
      `here <https://developer.nvidia.com/cuda-toolkit-archive>`__ Please note
      that we only support GPU acceleration on Linux platforms at the moment.

      Once CUDA® is properly installed, you only need to set a flag so the
      build system knows what to do:

      .. code:: sh

         AER_THRUST_BACKEND=CUDA

      For example,

      .. code:: sh

         qiskit-aer$ python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA

      See `here <https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md>`__ 
      for detailed GPU support information.

      .. raw:: html

         <h3>Building with MPI support</h3>

      Qiskit Aer can parallelize its simulation on the cluster systems by
      using MPI. This can extend available memory space to simulate quantum
      circuits with larger number of qubits and also can accelerate the
      simulation by parallel computing. To use MPI support, any MPI library
      (i.e. OpenMPI) should be installed and configured on the system.

      Qiskit Aer supports MPI both with and without GPU support. Currently
      following simulation methods are supported to be parallelized by MPI.

      -  statevector
      -  density_matrix
      -  unitary

      To enable MPI support, the following flag is needed for build system
      based on CMake.

      .. code:: sh

         AER_MPI=True

      For example,

      .. code:: sh

         qiskit-aer$ python ./setup.py bdist_wheel -- -DAER_MPI=True

      See `here <https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md>`__ 
      for detailed MPI support information.


Simulating your first quantum program with Qiskit Aer
=====================================================
Now that you have Qiskit Aer installed, you can start simulating a quantum circuit. 
Here is a basic example:

.. code:: python

  import qiskit
  from qiskit_aer.primitives import SamplerV2

  # Generate 3-qubit GHZ state
  circ = qiskit.QuantumCircuit(3)
  circ.h(0)
  circ.cx(0, 1)
  circ.cx(1, 2)
  circ.measure_all()

  # Construct an ideal simulator with SamplerV2
  sampler = SamplerV2()
  job = sampler.run([circ], shots=128)

  # Perform an ideal simulation
  result_ideal = job.result()
  counts_ideal = result_ideal[0].data.meas.get_counts()
  print('Counts(ideal):', counts_ideal)

Ready to get going?...
======================

.. raw:: html

   <div class="tutorials-callout-container">
      <div class="row">

.. qiskit-call-to-action-item::
   :description: Find out about Qiskit Aer
   :header: Dive into the tutorials
   :button_link:  ./tutorials/index.html
   :button_text: Qiskit Aer tutorials

.. raw:: html

      </div>
   </div>


.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
