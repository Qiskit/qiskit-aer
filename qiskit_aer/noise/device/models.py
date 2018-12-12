# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Simplified noise models for devices backends.
"""

from .parameters import readout_error_values
from .parameters import gate_time_values
from .parameters import gate_error_values
from .parameters import thermal_relaxation_values

from ..noise_model import NoiseModel
from ..readout_error import ReadoutError
from ..errors.standard_errors import depolarizing_error
from ..errors.standard_errors import thermal_relaxation_error


def depolarizing_noise_model(properties, standard_gates=True):
    """Generae a depolarizing noise model from device backend properties.

    Params:
        properties (BackendProperties): backend properties
        standard_gates (bool): If true return errors as standard
                               qobj gates. If false return as unitary
                               qobj instructions (Default: True)

    Returns:
        NoiseModel: An approximate noise model for the backend.

    Additional Information:

        The noise model will consist:
        * Single qubit readout errors on measurements.
        * Depolarizing errors on gates defined be defined in
          `BackendProperties.gates`.

        For best practice in simulating a backend make sure that the
        circuit is compiled using the set of basis gates in the noise
        module by setting:
            `basis_gates = noise_model.basis_gates`
        and using the device coupling map with:
            `coupling_map = backend.configuration().coupling_map`
    """
    noise_model = NoiseModel()

    # Add single-qubit readout errors
    for qubits, error in device_readout_errors(properties):
        noise_model.add_readout_error(error, qubits)

    # Add depolarizing gate errors
    gate_errors = device_depolarizing_errors(properties,
                                             standard_gates=standard_gates)
    for name, qubits, error in gate_errors:
        noise_model.add_quantum_error(error, name, qubits)

    return noise_model


def thermal_relaxation_noise_model(properties, gate_times=None):
    """Thermal relaxation noise model derived from backend properties.

    NOTE: gate_time field may be removed in the future when online
    backends return full gate time information.

    Params:
        properties (BackendProperties): backend properties
        gate_times (list): Override device gate times with custom
                           values. If None use gate times from
                           backend properties. (Default: None)

    Returns:
        NoiseModel: A noise model for the IBM backend containing.

    Additional Information:

        The noise model will consist:
        * Single qubit readout errors on measurements.
        * Single-qubit relaxation errors on all qubits participating in
          a noisy quantum gate. The relaxation strength is determined by
          the individual qubit T1 and T2 values and the gate_time of the
          gate as defined in `BackendProperties.gates`.

        Secifying custom gate times:
        If non-default values are used gate_times should be a list
        of tuples (name, qubits, value) where name is the gate name
        string, qubits is a list of qubits the gate acts on, and value
        is the gate time in nanoseconds. This replaces gate times
        extracted from backend properties.

        For best practice in simulating a backend make sure that the
        circuit is compiled using the set of basis gates in the noise
        module by setting:
            `basis_gates = noise_model.basis_gates`
        and using the device coupling map with:
            `coupling_map = backend.configuration().coupling_map`
    """

    noise_model = NoiseModel()

    # Add single-qubit readout errors
    for qubits, error in device_readout_errors(properties):
        noise_model.add_readout_error(error, qubits)

    # Add depolarizing gate errors
    gate_errors = device_thermal_relaxation_errors(properties)
    for name, qubits, error in gate_errors:
        noise_model.add_quantum_error(error, name, qubits)

    return noise_model


def device_depolarizing_errors(properties, standard_gates=True):
    """Get depolarizing noise quantum error objects for backend gates

    Args:
        properties (BackendProperties): device backend properties
        standard_gates (bool): If true return errors as standard
                               qobj gates. If false return as unitary
                               qobj instructions (Default: True)

    Returns:
        list: A list of tuples (name, qubits, error).
    """
    errors = []
    for name, qubits, value in gate_error_values(properties):
        if value is not None and value > 0:
            error = depolarizing_error(value, len(qubits),
                                       standard_gates=standard_gates)
            errors.append((name, qubits, error))
    return errors


def device_readout_errors(properties):
    """Get readout error objects for each qubit from backend properties

    Args:
        properties (BackendProperties): device backend properties

    Returns:
        list: A list of pairs (qubits, value) for qubits with non-zero
        readout error values.
    """
    errors = []
    for qubit, value in enumerate(readout_error_values(properties)):
        if value is not None and value > 0:
            probabilities = [[1 - value, value], [value, 1 - value]]
            errors.append(([qubit], ReadoutError(probabilities)))
    return errors


def device_thermal_relaxation_errors(properties, gate_times=None):
    """Get depolarizing noise quantum error objects for backend gates

    Args:
        properties (BackendProperties): device backend properties
        gate_times (list): Override device gate times with custom
                           values. If None use gate times from
                           backend properties. (Default: None)

    Returns:
        dict: A dictionary of pairs name: (qubits, error). If gate
        error information is not available None will be returned for
        value.

    Additional Information:
        If non-default values are used gate_times should be a list
        of tuples (name, qubits, value) where name is the gate name
        string, qubits is a list of qubits the gate acts on, and value
        is the gate time in nanoseconds.
    """
    gate_times = gate_time_values(properties)
    relax_values = thermal_relaxation_values(properties)
    errors = []
    for name, qubits, gate_time in gate_times:
        if gate_time is not None and gate_time > 0:
            # convert gate time to same units as T1 and T2 (microseconds)
            gate_time = gate_time / 1000
            # Construct a tensor product of single qubit relaxation errors
            # for any multi qubit gates
            first = True
            error = None
            for qubit in reversed(qubits):
                t1, t2, freq = relax_values[qubit]
                if first:
                    error = thermal_relaxation_error(t1, t2, gate_time)
                    first = False
                else:
                    error = error.kron(thermal_relaxation_error(t1, t2, gate_time))
            if error is not None:
                errors.append((name, qubits, error))
    return errors
