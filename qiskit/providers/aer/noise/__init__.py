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

"""Noise module for Qiskit Aer.

This module contains classes and functions to build a noise model for
simulating a Qiskit quantum circuit in the presence of errors.

Noise Models
------------
Noise models for a noisy simulator are represented using the `NoiseModel`
class. This can be used to generate a custom noise model, or an automatic
noise model can be generated for a device using built in functions.

Automatically Generated Noise Models
------------------------------------
Approximate noise models for a hardware device can be generated from the
device properties using the functions from the `device` submodule.

   Basic device noise model
   ------------------------
   Generates a noise mode based on 1 and 2 qubit gate errors consisting of
   a depolarizing error followed by a thermal relaxation error, and readout
   errors on measurement outcomes. The error parameters are tuned for each
   individual qubit based on the T_1, T_2, frequency and readout error
   parameters for each qubit, and the gate error and gate time parameters
   for each gate obtained from the device backend properties.

Custom Noise Models
-------------------
Custom noise models may be constructed by adding errors to a NoiseModel
object. Errors are represented using by the `QuantumError` and
`ReadoutError` classes from the `noise.errors` module:

   Quantum errors
   --------------
   Errors that affect the quantum state during a simulation. They may be
   applied after specific circuit gates or reset operations, or before
   measure operations of qubits.

   ReadoutError
   ------------
   Errors that apply to classical bit registers after a measurement. They
   do not change the quantum state of the system, only the recorded
   classical measurement outcome.

Helper functions exist for generating standard quantum error channels in
the `noise.errors` module. These allow simple generation of the follow
canonical types of quantum errors:

   Kraus error
   Mixed unitary error
   Coherent unitary error
   Pauli error
   Depolarizing error
   Reset error
   Thermal relaxation error
   Amplitude damping error
   Phase damping error
   Combined phase and amplitude damping error

Noise Utilities
--------------
The `noise.utils` module contains utilities for noise models and errors including:
   `remap_noise_model` for remapping the qubits specified in a NoiseModel.
   `approximate_quantum_error` for approximating a general quantum error via
   a provided set of errors (e.g. approximating amplitude damping via reset errors).
   `approximate_noise_model` for approximating all the errors in a nose model using
   the same provided set of errors.
"""

from .noise_model import NoiseModel
from . import errors
from . import device
from . import utils
