# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Functions to extract device error parameters from backend properties.
"""

from numpy import inf


def gate_error_values(properties):
    """Get gate error values for backend gate from backend properties

    Args:
        properties (BackendProperties): device backend properties

    Returns:
        list: A list of tuples (name, qubits, value). If gate
        error information is not available None will be returned for
        value.
    """
    values = []
    for gate in properties.gates:
        name = gate.gate
        qubits = gate.qubits
        value = _check_for_item(gate.parameters, 'gate_error')
        values.append((name, qubits, value))
    return values


def gate_time_values(properties):
    """Get gate time values for backend gate from backend properties

    Args:
        properties (BackendProperties): device backend properties

    Returns:
        list: A list of tuples (name, qubits, value). If gate
        time information is not available None will be returned for
        value.
    """
    values = []
    for gate in properties.gates:
        name = gate.gate
        qubits = gate.qubits
        value = _check_for_item(gate.parameters, 'gate_time')
        values.append((name, qubits, value))
    return values


def readout_error_values(properties):
    """Get readout error values for each qubit from backend properties

    Args:
        properties (BackendProperties): device backend properties

    Returns:
        list: A list of readout error values for qubits. If readout
        error information is not available None will be returned for value.
    """
    values = []
    for qubit, qubit_props in enumerate(properties.qubits):
        # Get the readout error value
        value = _check_for_item(qubit_props, 'readout_error')
        values.append(value)
    return values


def thermal_relaxation_values(properties):
    """Return list of T1, T2 and frequency values from backend properties.

    Args:
        properties (BackendProperties): device backend properties

    Returns:
        list: A list of tuples (T1, T2, freq) for each qubit in the device
        where T1 and T2 are in microsecond units, and frequency is in GHz.
        If T1, T2, of frequency cannot be loaded for qubit a value of
        Numpy.inf will be used.
    """
    values = []
    for qubit, qubit_props in enumerate(properties.qubits):
        # Get the readout error value
        t1 = _check_for_item(qubit_props, 'T1')
        t2 = _check_for_item(qubit_props, 'T2')
        freq = _check_for_item(qubit_props, 'frequency')
        if t1 is None:
            t1 = inf
        if t2 is None:
            t2 = inf
        if freq is None:
            freq = inf
        # NOTE: T2 cannot be larged than 2 * T1 for a physical noise
        # channel, however if a backend eroneously reports such a value we
        # truncated it here:
        t2 = min(2 * t1, t2)
        values.append((t1, t2, freq))
    return values


def _check_for_item(lst, name):
    """Search list for item with given name."""
    filtered = [item.value for item in lst if item.name == name]
    if len(filtered) == 0:
        return None
    else:
        return filtered[0]
