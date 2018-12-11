# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Noise module for Qiskit Aer.

This module contains classes and functions to build a noise model for
simulating a Qiskit quantum circuit in the presence of errors.

Noise Models
----------------------
Noise models for a noisy simulator are represented using the `NoiseModel`
class. This can be used to generate a custom noise model, or an automatic
noise model can be generated for a device using built in functions.

Automatically Generated Noise Models
------------------------------------
Approximate noise models for a hardware device can be generated from the
device properties using the functions from the `device` submodule.

    * `device.depolarizing_noise_model`: Generates a noise model based on
       one-qubit depolarizing errors acting after X90 pulses during u1, u2,
       and u3 gates, two-qubit depolarizing errors acting after cx gates,
       and readout errors acting after measurement. The error parameters
       are tuned for each individual qubit based on 1 and 2-qubit error
       parameters from the device backend properties.

    * `device.thermal_relaxation_noise_model`: Generates a noise mode
       based on one-qubit thermal relaxation errors sacting after X90
       pulses during u1, u2, and u3 gates, acting on both qubits after cx
       gates, and readout errors acting after measurement. The error
       parameters are tuned for each individual qubit based on the T_1,
       T_2, and single qubit gate time parameters from the device backend
       properties.

Custom Noise Models
-------------------
Custom noise models may be constructed by adding errors to a NoiseModel
object. Errors are represented using by the `QuantumError` and
`ReadoutError` classes:

    * `QuantumErrors`: Errors that affect the quantum state during a
       simulation. They may be applied after specific circuit gates or
       reset operations, or before measure operations of qubits.

    * `ReadoutErrors`: Errors that apply to classical bit registers
       after a measurement. They do not change the quantum state of the
       system, only the recorded classical measurement outcome.

Helper functions exist for generating standard quantum error channels in
the `errors` submodule. These functions are:

    * `errors.kraus_error`
    * `errors.mixed_unitary_error`
    * `errors.coherent_unitary_error`
    * `errors.pauli_error`
    * `errors.depolarizing_error`
    * `errors.thermal_relaxation_error`
    * `errors.phase_amplitude_damping_error`
    * `errors.amplitude_damping_error`
    * `errors.phase_damping_error`
"""

from .noise_model import NoiseModel
from .quantum_error import QuantumError
from .readout_error import ReadoutError
from . import errors
from . import device
