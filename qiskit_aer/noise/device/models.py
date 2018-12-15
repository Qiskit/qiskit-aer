# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Simplified noise models for devices backends.
"""

from numpy import inf, exp

from .parameters import readout_error_values
from .parameters import gate_param_values
from .parameters import thermal_relaxation_values

from ..noiseerror import NoiseError
from ..noise_model import NoiseModel
from ..errors.readout_error import ReadoutError
from ..errors.standard_errors import depolarizing_error
from ..errors.standard_errors import thermal_relaxation_error


def basic_device_noise_model(properties,
                             readout_error=True,
                             thermal_relaxation=True,
                             temperature=0,
                             gate_times=None,
                             standard_gates=True):
    """Approximate device noise model derived from backend properties.

    Params:
        properties (BackendProperties): backend properties
        readout_errors (Bool): Include readout errors in model
                               (Default: True).
        thermal_relaxation (Bool): Include thermal relaxation errors
                                   (Default: True).
        temperature (double): qubit temperature in milli-Kelvin (mK) for
                              thermal relaxation errors (Default: 0).
        gate_times (list): Custom gate times for thermal relaxation errors.
                           Used to extend or override the gate times in
                           the backend properties (Default: None).
        standard_gates (bool): If true return errors as standard
                               qobj gates. If false return as unitary
                               qobj instructions (Default: True)

    Returns:
        NoiseModel: An approximate noise model for the device backend.

    Additional Information:

        The noise model includes the following errors:

        If readout_error is True:
            * Single qubit readout errors on measurements.

        If thermal_relaxation is True:
            * Single-qubit gate errors consisting of a depolarizing error
              followed by a thermal relaxation error for the qubit the gate
              acts on.
            * Two-qubit gate errors consisting of a 2-qubit depolarizing
              error followed by single qubit thermal relaxation errors for
              all qubits participating in the gate.

        Else if thermal_relaxation is False:
            * Single-qubit depolarizing gate errors.
            * Multi-qubit depolarizing gate errors.

        For best practice in simulating a backend make sure that the
        circuit is compiled using the set of basis gates in the noise
        module by setting:
            `basis_gates = noise_model.basis_gates`
        and using the device coupling map with:
            `coupling_map = backend.configuration().coupling_map`

        Secifying custom gate times:

        The `gate_times` kwarg can be used to specify custom gate times
        to add gate errors using the T1 and T2 values from the backend
        properties. This should be passed as a list of tuples
            `gate_times=[(name, value), ...]`
        where `name` is the gate name string, and `value` is the gate time
        in nanoseconds.

        If a custom gate is specified that already exists in
        the backend properties, the `gate_times` value will override the
        gate time value from the backend properties.
        If non-default values are used gate_times should be a list
    """

    noise_model = NoiseModel()

    # Add single-qubit readout errors
    if readout_error:
        for qubits, error in basic_device_readout_errors(properties):
            noise_model.add_readout_error(error, qubits)

    # Add gate errors
    gate_errors = basic_device_gate_errors(properties,
                                           thermal_relaxation=thermal_relaxation,
                                           gate_times=gate_times,
                                           temperature=temperature,
                                           standard_gates=standard_gates)
    for name, qubits, error in gate_errors:
        noise_model.add_quantum_error(error, name, qubits)

    return noise_model


def basic_device_readout_errors(properties):
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


def basic_device_gate_errors(properties, thermal_relaxation=True,
                             gate_times=None, temperature=0,
                             standard_gates=True):
    """Get depolarizing noise quantum error objects for backend gates

    Args:
        properties (BackendProperties): device backend properties
        temperature (double): qubit temperature in milli-Kelvin (mK).
        gate_times (list): Override device gate times with custom
                           values. If None use gate times from
                           backend properties. (Default: None)

    Returns:
        dict: A dictionary of pairs name: (qubits, error). If gate
        error information is not available None will be returned for
        value.

    Additional Information:
        If non-default values are used gate_times should be a list
        of tuples (name, qubits, value) where name is the gate name string,
        qubits is a list of qubits or None to apply gate time to this
        gate one any set of qubits, and value is the gate time in
        nanoseconds.
    """
    # Generate custom gate time dict
    custom_times = {}
    if thermal_relaxation:
        # If including thermal relaxation errors load
        # T1, T2, and frequency values from properties
        relax_params = thermal_relaxation_values(properties)
        # If we are specifying custom gate times include
        # them in the custom times dict
        if gate_times:
            for name, qubits, value in gate_times:
                if name in custom_times:
                    custom_times[name].append((qubits, value))
                else:
                    custom_times[name] = [(qubits, value)]
    # Get the device gate parameters from properties
    device_gate_params = gate_param_values(properties)

    # Construct quantum errors
    errors = []
    for name, qubits, gate_time, gate_error in device_gate_params:
        # Check for custom gate time
        relax_time = gate_time
        # Override with custom value
        if name in custom_times:
            filtered = [val for q, val in custom_times[name]
                        if q is None or q == qubits]
            if filtered:
                # get first value
                relax_time = filtered[0]
        # Get depolarizing error channel
        depol_error = _device_depolarizing_error(qubits, gate_error,
                                                 relax_time,
                                                 relax_params,
                                                 temperature,
                                                 thermal_relaxation,
                                                 standard_gates)
        # Get relaxation error
        relax_error = _device_thermal_relaxation_error(qubits, relax_time,
                                                       relax_params,
                                                       temperature,
                                                       thermal_relaxation)
        # Combine errors
        if depol_error is None and relax_error is None:
            # No error for this gate
            pass
        elif depol_error is not None and relax_error is None:
            # Append only the depolarizing error
            errors.append((name, qubits, depol_error))
            # Append only the relaxation error
        elif relax_error is not None and depol_error is None:
            errors.append((name, qubits, relax_error))
        else:
            # Append a combined error of depolarizing error
            # followed by a relaxation error
            combined_error = depol_error.compose(relax_error)
            errors.append((name, qubits, combined_error))
    return errors


def _device_depolarizing_error(qubits, gate_error, gate_time, relax_params,
                               temperature, thermal_relaxation=True,
                               standard_gates=True):
    """Construct a depolarizing_error for device"""
    error = None
    if not thermal_relaxation:
        # Model gate error entirely as depolarizing error
        p_depol = _depol_error_value_one_qubit(gate_error)
    else:
        # Model gate error as thermal relaxation and depolarizing
        # error.
        # Get depolarizing probability
        if len(qubits) == 1:
            t1, t2, _ = relax_params[qubits[0]]
            p_depol = _depol_error_value_one_qubit(gate_error,
                                                   gate_time,
                                                   t1=t1, t2=t2)
        elif len(qubits) == 2:
            q0_t1, q0_t2, _ = relax_params[qubits[0]]
            q1_t1, q1_t2, _ = relax_params[qubits[1]]
            p_depol = _depol_error_value_two_qubit(gate_error,
                                                   gate_time,
                                                   qubit0_t1=q0_t1,
                                                   qubit0_t2=q0_t2,
                                                   qubit1_t1=q1_t1,
                                                   qubit1_t2=q1_t2)
        else:
            raise NoiseError("Device noise model only supports" +
                             "1 and 2-qubit gates when using "
                             "thermal_relaxation=True.")
    if p_depol > 0:
        error = depolarizing_error(p_depol, len(qubits),
                                   standard_gates=standard_gates)
    return error


def _device_thermal_relaxation_error(qubits, gate_time, relax_params,
                                     temperature, thermal_relaxation=True):
    """Construct a thermal_relaxation_error for device"""
    # Check trivial case
    if not thermal_relaxation or gate_time is None or gate_time == 0:
        return None
    # convert gate time to same units as T1 and T2 (microseconds)
    gate_time = gate_time / 1000
    # Construct a tensor product of single qubit relaxation errors
    # for any multi qubit gates
    first = True
    error = None
    for qubit in reversed(qubits):
        t1, t2, freq = relax_params[qubit]
        population = _excited_population(freq, temperature)
        if first:
            error = thermal_relaxation_error(t1, t2, gate_time,
                                             population)
            first = False
        else:
            single = thermal_relaxation_error(t1, t2, gate_time,
                                              population)
            error = error.kron(single)
    return error


def _excited_population(freq, temperature):
    """Return excited state population"""
    population = 0
    if freq != inf and temperature != 0:
        # Compute the excited state population from qubit
        # frequency and temperature
        # Boltzman constant  kB = 6.62607015e-34 (eV/K)
        # Planck constant h =  6.62607015e-34 (eV.s)
        # qubit temperature temperatue = T (mK)
        # qubit frequency frequency = f (GHz)
        # excited state population = 1/(1+exp((2hf*1e9)/(kbT*1e-3)))
        exp_param = exp((95.9849 * freq) / abs(temperature))
        population = 1 / (1 + exp_param)
        if temperature < 0:
            # negative temperate implies |1> is thermal ground
            population = 1 - population
    return population


def _depol_error_value_one_qubit(gate_error, gate_time=0, t1=inf, t2=inf):
    """Return 2-qubit depolarizing channel probability for device model"""
    # Check trivial case where there is no gate error
    if gate_error is None:
        return None
    if gate_error == 0:
        return 0

    # Check t1 and t2 are valid
    if t1 <= 0:
        raise NoiseError("Invalid T_1 relaxation time parameter: T_1 <= 0.")
    if t2 <= 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 <= 0.")
    if t2 - 2 * t1 > 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.")

    # If T1 or T2 we have only a depolarizing error model
    # in this case p_depol = dim * gate_error / (dim - 1)
    # with dim = 2 for 1-qubit
    if gate_time is None:
        gate_time = 0
    if gate_time == 0 or (t1 == inf and t2 == inf):
        if gate_error is not None and gate_error > 0:
            return 2 * gate_error
        else:
            return 0

    # Otherwise we calculate the depolarizing error probability to account
    # for the difference between the relaxation error and gate error
    if t1 == inf:
        par1 = 1
    else:
        par1 = exp(-gate_time / t1)
    if t2 == inf:
        par2 = 1
    else:
        par2 = exp(-gate_time / t2)
    p_depol = 1 + 3 * (2 * gate_error - 1) / (par1 + 2 * par2)
    return p_depol


def _depol_error_value_two_qubit(gate_error, gate_time=0,
                                 qubit0_t1=inf, qubit0_t2=inf,
                                 qubit1_t1=inf, qubit1_t2=inf):
    """Return 2-qubit depolarizing channel probability for device model"""
    # Check trivial case where there is no gate error
    if gate_error is None:
        return None
    if gate_error == 0:
        return 0

    # Check t1 and t2 are valid
    if qubit0_t1 <= 0 or qubit1_t1 <= 0:
        raise NoiseError("Invalid T_1 relaxation time parameter: T_1 <= 0.")
    if qubit0_t2 <= 0 or qubit1_t2 <= 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 <= 0.")
    if qubit0_t2 - 2 * qubit0_t1 > 0 or qubit1_t2 - 2 * qubit1_t1 > 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.")

    # If T1 or T2 we have only a depolarizing error model
    # in this case p_depol = dim * gate_error / (dim - 1)
    # with dim = 4 for 2-qubits
    if gate_time is None:
        gate_time = 0
    if gate_time == 0 or (qubit0_t1 == inf and qubit0_t2 == inf and
                          qubit1_t1 == inf and qubit1_t2 == inf):
        if gate_error is not None and gate_error > 0:
            return 4 * gate_error / 3
        else:
            return 0

    # Otherwise we calculate the depolarizing error probability to account
    # for the difference between the relaxation error and gate error
    if qubit0_t1 == inf:
        q0_par1 = 1
    else:
        q0_par1 = exp(-gate_time / qubit0_t1)
    if qubit0_t2 == inf:
        q0_par2 = 1
    else:
        q0_par2 = exp(-gate_time / qubit0_t2)
    if qubit1_t1 == inf:
        q1_par1 = 1
    else:
        q1_par1 = exp(-gate_time / qubit1_t1)
    if qubit1_t2 == inf:
        q1_par2 = 1
    else:
        q1_par2 = exp(-gate_time / qubit1_t2)
    denom = (q0_par1 + q1_par1 + q0_par1 * q1_par1 +
             4 * q0_par2 * q1_par2 +
             2 * (q0_par2 + q1_par2) +
             2 * (q1_par1 * q0_par2 + q0_par1 * q1_par2))
    p_depol = 1 + (5 / 3) * (4 * gate_error - 3) / denom
    return p_depol
