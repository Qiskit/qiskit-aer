# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Simplified noise models for devices backends.
"""

from ..noise_model import NoiseModel
from ..readout_error import ReadoutError
from ..errors.standard_errors import depolarizing_error
from ..errors.standard_errors import thermal_relaxation_error


def depolarizing_noise_model(properties, standard_gates=True):
    """Depolarizing noise model derived from backend properties.

    Params:
        properties (BackendProperties): backend properties
        standard_gates (bool): If true return errors as standard
                               qobj gates. If false return as unitary
                               qobj instructions (Default: True)

    Returns:
        NoiseModel: A noise model for the IBM backend containing.

    Additional Information:
        Make sure qobj is compiled with:

        * basis_gates = 'u1,u2,u3,cx',
        * coupling_map = backend.configuration().coupling_map

        The noise model is based on the current device calibration
        properties. These parameters are used to construct a simplified
        noise model for the device consisting of:

        * Single qubit depolarizing errors on u1, u2, u3 gates implemented
          as X-90 pulses with Z-rotations.
        * Single qubit readout errors on measurements.
        * Two qubit depolarizing errors on cx gates.
    """
    noise_model = NoiseModel()
    noise_model.set_x90_single_qubit_gates(['u1', 'u2', 'u3'])

    # Add 1-qubit gate errors
    gate_errors1 = one_qubit_depolarizing_errors(properties,
                                                 standard_gates=standard_gates)
    for qubits, error in gate_errors1:
        noise_model.add_quantum_error(error, 'x90', qubits)

    # Add 2-qubit gate errors
    gate_errors2 = two_qubit_depolarizing_errors(properties,
                                                 standard_gates=standard_gates)
    for qubits, error in gate_errors2:
        noise_model.add_quantum_error(error, 'cx', qubits)

    # Add readout errors
    for qubits, error in one_qubit_readout_errors(properties):
        noise_model.add_readout_error(error, qubits)

    return noise_model


def thermal_relaxation_noise_model(properties, gate_time_cx_ns):
    """Thermal relaxation noise model derived from backend properties.

    Params:
        properties (BackendProperties): backend properties
        gate_time_cx_ns (double): The gate time in nanoseconds to use for
                                  two-qubit CX gates.

    Returns:
        NoiseModel: A noise model for the IBM backend containing.

    Additional Information:
        Make sure qobj is compiled with:

        * basis_gates = 'u1,u2,u3,cx',
        * coupling_map = backend.configuration().coupling_map

        The noise model is based on the current device calibration
        properties. These parameters are used to construct a simplified
        noise model for the device consisting of:

        * Single qubit thermal relaxation errors on u1, u2, u3 gates
          implemented as X-90 pulses with Z-rotations.
        * Two qubit thermal relaxation errors on cx gates.
        * Single qubit readout errors on measurements.

        Note that this simplified noise model does not simulate relaxation
        on idle qubits during gates.
    """
    noise_model = NoiseModel()
    noise_model.set_x90_single_qubit_gates(['u1', 'u2', 'u3'])

    # Add 1-qubit gate errors
    gate_errors1 = one_qubit_thermal_relaxation_errors(properties)
    for qubits, error in gate_errors1:
        noise_model.add_quantum_error(error, 'x90', qubits)

    # Add 2-qubit gate errors
    gate_errors2 = two_qubit_thermal_relaxation_errors(properties, gate_time_cx_ns)
    for qubits, error in gate_errors2:
        noise_model.add_quantum_error(error, 'cx', qubits)

    # Add readout errors
    for qubits, error in one_qubit_readout_errors(properties):
        noise_model.add_readout_error(error, qubits)

    return noise_model


def one_qubit_readout_errors(properties):
    """Return a list of single-qubit ReadoutErrors.

    The error parameters are derived from the BackendProperties
    of a backend.

    Params:
        properties (BackendProperties): backend properties

    Returns
        list[(qubits, ReadoutError)]: a list of pairs of qubits
        and ReadoutError objects.
    """
    error_pairs = []
    for params in _one_qubit_error_params(properties):
        qubits = params['qubits']
        val = params['readout_error']
        error = ReadoutError([[1 - val, val],
                              [val, 1 - val]])
        error_pairs.append((qubits, error))
    return error_pairs


def one_qubit_depolarizing_errors(properties, standard_gates=True):
    """Return a list of single-qubit depolarizing errors.

    The error parameters are derived from the BackendProperties
    of a backend.

    Params:
        properties (BackendProperties): backend properties
        standard_gates (bool): If true return errors as standard
                               qobj gates. If false return as unitary
                               qobj instructions (Default: True)

    Returns
        list[(qubits, QuantumError)]: a list of pairs of qubits
        and QuantumError objects.
    """
    error_pairs = []
    for params in _one_qubit_error_params(properties):
        qubits = params['qubits']
        val = params['gate_error']
        error = depolarizing_error(val, 1, standard_gates=standard_gates)
        error_pairs.append((qubits, error))
    return error_pairs


def one_qubit_thermal_relaxation_errors(properties):
    """Return a list of single-qubit thermal relaxaton errors.

    The error parameters are derived from the BackendProperties
    of a backend.

    Params:
        properties (BackendProperties): backend properties

    Returns
        list[(qubits, QuantumError)]: a list of pairs of qubits
        and QuantumError objects.
    """
    error_pairs = []
    for params in _one_qubit_error_params(properties):
        qubits = params['qubits']
        t1 = params['t1']
        t2 = params['t2']
        gate_time = params['gate_time'] + params['buffer']
        error = thermal_relaxation_error(t1, t2, gate_time)
        error_pairs.append((qubits, error))
    return error_pairs


def two_qubit_depolarizing_errors(properties, standard_gates=True):
    """Return a list of two-qubit depolarizing errors.

    The error parameters are derived from the BackendProperties
    of a device backend.

    Params:
        properties (BackendProperties): backend properties
        standard_gates (bool): If true return errors as standard
                               qobj gates. If false return as unitary
                               qobj instructions (Default: True)

    Returns
        list[(qubits, QuantumError)]: a list of pairs of qubits
        and QuantumError objects.
    """
    error_pairs = []
    for params in _two_qubit_error_params(properties):
        qubits = params['qubits']
        val = params['gate_error']
        error = depolarizing_error(val, 2,
                                   standard_gates=standard_gates)
        error_pairs.append((qubits, error))
    return error_pairs


def two_qubit_thermal_relaxation_errors(properties, gate_time_ns):
    """Return a list of two-qubit thermal relaxaton errors.

    The error parameters are derived from the BackendProperties
    of a backend.

    Params:
        properties (BackendProperties): backend properties
        gate_time_ns (double): CX gate time in nanoseconds.

    Returns
        list[(qubits, QuantumError)]: a list of pairs of qubits
        and QuantumError objects.
    """
    params1 = _one_qubit_error_params(properties)
    error_pairs = []
    for params2 in _two_qubit_error_params(properties):
        qubits = params2['qubits']
        params_q0 = params1[qubits[0]]
        params_q1 = params1[qubits[1]]
        error0 = thermal_relaxation_error(params_q0['t1'],
                                          params_q0['t2'],
                                          gate_time_ns)
        error1 = thermal_relaxation_error(params_q1['t1'],
                                          params_q1['t2'],
                                          gate_time_ns)
        error_pairs.append((qubits, error1.kron(error0)))
    return error_pairs


def _check_for_item(lst, name):
    """Search list for item with given name."""
    filtered = [item.value for item in lst if item.name == name]
    if len(filtered) == 0:
        return None
    else:
        return filtered[0]


def _two_qubit_error_params(properties):
    """Return error parameters of CX gates from backend properties."""
    # Add 2-qubit gate errors
    error_params = []
    for gate2 in properties.gates:
        gate_error = _check_for_item(gate2.parameters, 'gateerr')
        if gate_error is not None:
            params = {
                'qubits': [int(q) for q in gate2.qubits],
                'gate_error': gate_error
            }
            error_params.append(params)
    return error_params


def _one_qubit_error_params(properties):
    """Return error parameters of X90 gates from backend properties."""
    # Add 2-qubit gate errors
    error_params = []
    for qubit, qubit_props in enumerate(properties.qubits):
        params = {
            'qubits': [qubit],
            'gate_error': _check_for_item(qubit_props, 'gateError'),
            'readout_error': _check_for_item(qubit_props, 'readoutError'),
            't1': _check_for_item(qubit_props, 'T1'),
            't2': _check_for_item(qubit_props, 'T2'),
            'frequency': _check_for_item(qubit_props, 'frequency'),
            'gate_time': _check_for_item(qubit_props, 'gateTime'),
            'buffer': _check_for_item(qubit_props, 'buffer')
        }
        # convert us to ns in T1 for common time units
        if params['t1'] is not None:
            params['t1'] *= 1e3
        if params['t2'] is not None:
            params['t2'] *= 1e3
        # NOTE: This is a work around since some backends currently
        # report T2 values greater than the allowed maximum of 2 * T1
        params['t2'] = min(2 * params['t1'], params['t2'])
        error_params.append(params)
    return error_params
