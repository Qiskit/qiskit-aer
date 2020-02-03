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

# pylint: disable=invalid-name
"""
Simplified noise models for devices backends.
"""

from numpy import inf, exp, allclose

from .parameters import readout_error_values
from .parameters import gate_param_values
from .parameters import thermal_relaxation_values

from ..noiseerror import NoiseError
from ..errors.readout_error import ReadoutError
from ..errors.standard_errors import depolarizing_error
from ..errors.standard_errors import thermal_relaxation_error


def basic_device_readout_errors(properties):
    """
    Return readout error parameters from a devices BackendProperties.

    Args:
        properties (BackendProperties): device backend properties

    Returns:
        list: A list of pairs ``(qubits, ReadoutError)`` for qubits with
        non-zero readout error values.
    """
    errors = []
    for qubit, value in enumerate(readout_error_values(properties)):
        if value is not None and not allclose(value, [0, 0]):
            probabilities = [[1 - value[0], value[0]], [value[1], 1 - value[1]]]
            errors.append(([qubit], ReadoutError(probabilities)))
    return errors


def basic_device_gate_errors(properties,
                             gate_error=True,
                             thermal_relaxation=True,
                             gate_lengths=None,
                             temperature=0,
                             standard_gates=True):
    """
    Return QuantumErrors derived from a devices BackendProperties.

    If non-default values are used gate_lengths should be a list
    of tuples ``(name, qubits, value)`` where ``name`` is the gate
    name string, ``qubits`` is either a list of qubits or ``None``
    to apply gate time to this gate one any set of qubits,
    and ``value`` is the gate time in nanoseconds.

    Args:
        properties (BackendProperties): device backend properties
        gate_error (bool): Include depolarizing gate errors (Default: True).
        thermal_relaxation (Bool): Include thermal relaxation errors
                                   (Default: True).
        gate_lengths (list): Override device gate times with custom
                             values. If None use gate times from
                             backend properties. (Default: None).
        temperature (double): qubit temperature in milli-Kelvin (mK)
                              (Default: 0).
        standard_gates (bool): If true return errors as standard
                               qobj gates. If false return as unitary
                               qobj instructions (Default: True).

    Returns:
        list: A list of tuples ``(label, qubits, QuantumError)``, for gates
        with non-zero quantum error terms, where `label` is the label of the
        noisy gate, `qubits` is the list of qubits for the gate.
    """
    # Initilize empty errors
    depol_error = None
    relax_error = None
    # Generate custom gate time dict
    custom_times = {}
    relax_params = []
    if thermal_relaxation:
        # If including thermal relaxation errors load
        # T1, T2, and frequency values from properties
        relax_params = thermal_relaxation_values(properties)
        # If we are specifying custom gate times include
        # them in the custom times dict
        if gate_lengths:
            for name, qubits, value in gate_lengths:
                if name in custom_times:
                    custom_times[name].append((qubits, value))
                else:
                    custom_times[name] = [(qubits, value)]
    # Get the device gate parameters from properties
    device_gate_params = gate_param_values(properties)

    # Construct quantum errors
    errors = []
    for name, qubits, gate_length, error_param in device_gate_params:
        # Check for custom gate time
        relax_time = gate_length
        # Override with custom value
        if name in custom_times:
            filtered = [
                val for q, val in custom_times[name]
                if q is None or q == qubits
            ]
            if filtered:
                # get first value
                relax_time = filtered[0]
        # Get depolarizing error channel
        if gate_error:
            depol_error = _device_depolarizing_error(
                qubits, error_param, relax_time, relax_params,
                thermal_relaxation, standard_gates)
        # Get relaxation error
        if thermal_relaxation:
            relax_error = _device_thermal_relaxation_error(
                qubits, relax_time, relax_params, temperature,
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


def _device_depolarizing_error(qubits,
                               error_param,
                               gate_time,
                               relax_params,
                               thermal_relaxation=True,
                               standard_gates=True):
    """Construct a depolarizing_error for device"""

    # We now deduce the depolarizing channel error parameter in the
    # presence of T1/T2 thermal relaxation. We assume the gate error
    # parameter is given by e = 1 - F where F is the average gate fidelity,
    # and that this average gate fidelity is for the composition
    # of a T1/T2 thermal relaxation channel and a depolarizing channel.

    # For the n-qubit depolarizing channel E_dep = (1-p) * I + p * D, where
    # I is the identity channel and D is the completely depolarizing
    # channel, the average gate fidelity is given by:
    # F(E_dep) = (1 - p) * F(I) + p * F(D)
    #          = (1 - p) * 1 + p * (1 / dim)
    # where F(I) = 1, F(D) = 1 / dim = 1 - p * (dim - 1) / dim
    # Hence we have that
    # p = (1 - F(E_dep)) / ((dim - 1) / dim)
    #   = dim * (1 - F(E_dep)) / (dim - 1)
    #   = dim * e / (dim - 1)
    # therefore p = dim * error_param / (dim - 1)
    # with dim = 2 ** N for an N-qubit gate error.

    error = None
    num_qubits = len(qubits)
    dim = 2 ** num_qubits

    if not thermal_relaxation:
        # Model gate error entirely as depolarizing error
        if error_param is not None and error_param > 0:
            dim = 2 ** num_qubits
            depol_param = dim * error_param / (dim - 1)
        else:
            depol_param = 0
    else:
        # Model gate error as thermal relaxation and depolarizing
        # error.
        # Get depolarizing probability
        if num_qubits == 1:
            t1, t2, _ = relax_params[qubits[0]]
            depol_param = _depol_error_value_one_qubit(
                error_param, gate_time, t1=t1, t2=t2)
        elif num_qubits == 2:
            q0_t1, q0_t2, _ = relax_params[qubits[0]]
            q1_t1, q1_t2, _ = relax_params[qubits[1]]
            depol_param = _depol_error_value_two_qubit(
                error_param,
                gate_time,
                qubit0_t1=q0_t1,
                qubit0_t2=q0_t2,
                qubit1_t1=q1_t1,
                qubit1_t2=q1_t2)
        else:
            raise NoiseError("Device noise model only supports "
                             "1 and 2-qubit gates when using "
                             "thermal_relaxation=True.")
    if depol_param > 0:
        # If the device reports an error_param greater than the maximum
        # allowed for a depolarzing error model we will get a non-physical
        # depolarizing parameter.
        # In this case we truncate it to 1 so that the error channel is a
        # completely depolarizing channel E(rho) = id / d
        depol_param = min(depol_param, 1.0)
        error = depolarizing_error(
            depol_param, num_qubits, standard_gates=standard_gates)
    return error


def _device_thermal_relaxation_error(qubits,
                                     gate_time,
                                     relax_params,
                                     temperature,
                                     thermal_relaxation=True):
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
    for qubit in qubits:
        t1, t2, freq = relax_params[qubit]
        population = _excited_population(freq, temperature)
        if first:
            error = thermal_relaxation_error(t1, t2, gate_time, population)
            first = False
        else:
            single = thermal_relaxation_error(t1, t2, gate_time, population)
            error = error.expand(single)
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


def _depol_error_value_one_qubit(error_param, gate_time=0, t1=inf, t2=inf):
    """Return 2-qubit depolarizing channel parameter for device model"""
    # Check trivial case where there is no gate error
    if error_param is None:
        return None
    if error_param == 0:
        return 0

    # Check t1 and t2 are valid
    if t1 <= 0:
        raise NoiseError("Invalid T_1 relaxation time parameter: T_1 <= 0.")
    if t2 <= 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 <= 0.")
    if t2 - 2 * t1 > 0:
        raise NoiseError(
            "Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.")

    if gate_time is None:
        gate_time = 0
    if gate_time == 0 or (t1 == inf and t2 == inf):
        if error_param is not None and error_param > 0:
            return 2 * error_param
        else:
            return 0

    # Otherwise we calculate the depolarizing error parameter to account
    # for the difference between the relaxation error and gate error
    if t1 == inf:
        par1 = 1
    else:
        par1 = exp(-gate_time / t1)
    if t2 == inf:
        par2 = 1
    else:
        par2 = exp(-gate_time / t2)
    depol_param = 1 + 3 * (2 * error_param - 1) / (par1 + 2 * par2)
    return depol_param


def _depol_error_value_two_qubit(error_param,
                                 gate_time=0,
                                 qubit0_t1=inf,
                                 qubit0_t2=inf,
                                 qubit1_t1=inf,
                                 qubit1_t2=inf):
    """Return 2-qubit depolarizing channel parameter for device model"""
    # Check trivial case where there is no gate error
    if error_param is None:
        return None
    if error_param == 0:
        return 0

    # Check t1 and t2 are valid
    if qubit0_t1 <= 0 or qubit1_t1 <= 0:
        raise NoiseError("Invalid T_1 relaxation time parameter: T_1 <= 0.")
    if qubit0_t2 <= 0 or qubit1_t2 <= 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 <= 0.")
    if qubit0_t2 - 2 * qubit0_t1 > 0 or qubit1_t2 - 2 * qubit1_t1 > 0:
        raise NoiseError(
            "Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.")

    if gate_time is None:
        gate_time = 0
    if gate_time == 0 or (qubit0_t1 == inf and
                          qubit0_t2 == inf and
                          qubit1_t1 == inf and
                          qubit1_t2 == inf):
        if error_param is not None and error_param > 0:
            return 4 * error_param / 3
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
    denom = (
        q0_par1 + q1_par1 + q0_par1 * q1_par1 + 4 * q0_par2 * q1_par2 +
        2 * (q0_par2 + q1_par2) + 2 * (q1_par1 * q0_par2 + q0_par1 * q1_par2))
    depol_param = 1 + 5 * (4 * error_param - 3) / denom
    return depol_param
