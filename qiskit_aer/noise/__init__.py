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

r"""
================================================
Noise Models (:mod:`qiskit_aer.noise`)
================================================

.. currentmodule:: qiskit_aer.noise

This module contains classes and functions to build a noise model for
simulating a Qiskit quantum circuit in the presence of errors.


Building Noise Models
=====================

The :class:`NoiseModel` class is used to represent noise model for the
:class:`~qiskit_aer.QasmSimulator`. It can be used to construct
custom noise models for simulator, to automatically generate a basic
device noise model for an IBMQ or fake backend.


Device Noise Models
-------------------

A simplified approximate :class:`NoiseModel` can be generated automatically
from the properties of real device backends from the IBMQ provider or
fake backends of the `fake_provider` using the :meth:`NoiseModel.from_backend`
method. See the method documentation for details.


**Example: Basic device noise model**

.. code-block:: python

    from qiskit import IBMQ
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.visualization import plot_histogram
    from qiskit_aer.noise import NoiseModel

    # Make a circuit
    circ = QuantumCircuit(3, 3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.measure([0, 1, 2], [0, 1, 2])

    # Get the noise model of ibmq_lima
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend_lima = provider.get_backend('ibmq_lima')
    noise_model = NoiseModel.from_backend(backend_lima)

    # Get coupling map from backend
    coupling_map = backend_lima.configuration().coupling_map

    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates

    # Perform a noise simulation
    backend = AerSimulator(noise_model=noise_model,
                           coupling_map=coupling_map,
                           basis_gates=basis_gates)
    transpiled_circuit = transpile(circ, backend)
    result = backend.run(transpiled_circuit).result()

    counts = result.get_counts(0)
    plot_histogram(counts)


**Example: Basic device noise model using a `fake_provider` backend**

.. code-block:: python

    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.visualization import plot_histogram
    from qiskit_aer.noise import NoiseModel
    from qiskit.providers.fake_provider import FakeVigo

    # Build noise model from backend properties
    backend = FakeVigo()
    noise_model = NoiseModel.from_backend(backend)

    # Get coupling map from backend
    coupling_map = backend.configuration().coupling_map

    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates

    # Make a circuit
    circ = QuantumCircuit(3, 3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.measure([0, 1, 2], [0, 1, 2])

    # Perform a noise simulation
    backend = AerSimulator(noise_model=noise_model,
                           coupling_map=coupling_map,
                           basis_gates=basis_gates)
    transpiled_circuit = transpile(circ, backend)
    result = backend.run(transpiled_circuit).result()

    counts = result.get_counts(0)
    plot_histogram(counts)


Custom Noise Models
-------------------

Custom noise models can be used by adding :class:`QuantumError` to circuit
gate, reset or measure instructions, and :class:`ReadoutError` to measure
instructions. This module includes several helper functions for generating
:class:`QuantumError` instances based on canonical error models used in
Quantum Information Theory that can simplify building noise models. See the
documentation for the :class:`NoiseModel` class for additional details.

**Example: depolarizing noise model**

.. code-block:: python

    from qiskit import QuantumCircuit, transpile, Aer
    from qiskit.visualization import plot_histogram
    from qiskit_aer import AerSimulator
    import qiskit_aer.noise as noise

    # Error probabilities
    prob_1 = 0.001  # 1-qubit gate
    prob_2 = 0.01   # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates

    # Make a circuit
    circ = QuantumCircuit(3, 3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.measure([0, 1, 2], [0, 1, 2])

    # Perform a noise simulation
    backend = AerSimulator(noise_model=noise_model,
                           coupling_map=coupling_map,
                           basis_gates=basis_gates)
    transpiled_circuit = transpile(circ, backend)
    result = backend.run(transpiled_circuit).result()

    counts = result.get_counts(0)
    plot_histogram(counts)


Classes
=======

The following are the classes used to represented noise and error terms.

.. autosummary::
    :toctree: ../stubs/

    NoiseModel
    QuantumError
    PauliLindbladError
    ReadoutError


Quantum Error Functions
=======================

The following functions can be used to generate many common types of
:class:`QuantumError` objects for inclusion in a :class:`NoiseModel`.

.. autosummary::
    :toctree: ../stubs/

    pauli_error
    depolarizing_error
    mixed_unitary_error
    coherent_unitary_error
    reset_error
    amplitude_damping_error
    phase_damping_error
    phase_amplitude_damping_error
    thermal_relaxation_error
    kraus_error


Noise Transpiler Passes
=======================

These transpiler passes can be used to build noise models that can be applied
to circuits via transpilation.

.. autosummary::
    :toctree: ../stubs/

    LocalNoisePass
    RelaxationNoisePass


Device Noise Parameters
=======================

The following are utility functions which can be used for extracting error
parameters and error objects from device `BackendProperties`.

.. autosummary::
    :toctree: ../stubs/

    device.basic_device_readout_errors
    device.basic_device_gate_errors
    device.gate_param_values
    device.gate_error_values
    device.gate_length_values
    device.readout_error_values
    device.thermal_relaxation_values
"""

# Noise and Error classes
from .noise_model import NoiseModel
from .errors import QuantumError
from .errors import PauliError
from .errors import PauliLindbladError
from .errors import ReadoutError

# Error generating functions
from .errors import kraus_error
from .errors import mixed_unitary_error
from .errors import coherent_unitary_error
from .errors import pauli_error
from .errors import depolarizing_error
from .errors import reset_error
from .errors import thermal_relaxation_error
from .errors import phase_amplitude_damping_error
from .errors import amplitude_damping_error
from .errors import phase_damping_error

# Transpiler Passes
from .passes import LocalNoisePass
from .passes import RelaxationNoisePass

# Submodules
from . import errors
from . import device
