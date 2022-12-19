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

import logging
from warnings import warn, catch_warnings, filterwarnings

from numpy import inf, exp, allclose

import qiskit.quantum_info as qi
from qiskit.circuit import Gate, Measure
from .parameters import _NANOSECOND_UNITS
from .parameters import gate_param_values
from .parameters import readout_error_values
from .parameters import thermal_relaxation_values
from ..errors.readout_error import ReadoutError
from ..errors.standard_errors import depolarizing_error
from ..errors.standard_errors import thermal_relaxation_error
from ..noiseerror import NoiseError

logger = logging.getLogger(__name__)


def basic_device_readout_errors(properties=None, target=None):
    """
    Return readout error parameters from either of device Target or BackendProperties.

    If ``target`` is supplied, ``properties`` will be ignored.

    Args:
        properties (BackendProperties): device backend properties
        target (Target): device backend target

    Returns:
        list: A list of pairs ``(qubits, ReadoutError)`` for qubits with
        non-zero readout error values.

    Raises:
        NoiseError: if neither properties nor target is supplied.
    """
    errors = []
    if target is None:
        if properties is None:
            raise NoiseError("Either properties or target must be supplied.")
        # create from BackendProperties
        for qubit, value in enumerate(readout_error_values(properties)):
            if value is not None and not allclose(value, [0, 0]):
                probabilities = [[1 - value[0], value[0]], [value[1], 1 - value[1]]]
                errors.append(([qubit], ReadoutError(probabilities)))
    else:
        # create from Target
        for q in range(target.num_qubits):
            meas_props = target.get("measure", None)
            if meas_props is None:
                continue
            prop = meas_props.get((q,), None)
            if prop is None:
                continue
            if hasattr(prop, "prob_meas1_prep0") and hasattr(prop, "prob_meas0_prep1"):
                p0m1, p1m0 = prop.prob_meas1_prep0, prop.prob_meas0_prep1
            else:
                p0m1, p1m0 = prop.error, prop.error
            probabilities = [[1 - p0m1, p0m1], [p1m0, 1 - p1m0]]
            errors.append(([q], ReadoutError(probabilities)))

    return errors


def basic_device_gate_errors(properties=None,
                             gate_error=True,
                             thermal_relaxation=True,
                             gate_lengths=None,
                             gate_length_units='ns',
                             temperature=0,
                             standard_gates=None,
                             warnings=None,
                             target=None):
    """
    Return QuantumErrors derived from either of a devices BackendProperties or Target.

    If non-default values are used gate_lengths should be a list
    of tuples ``(name, qubits, value)`` where ``name`` is the gate
    name string, ``qubits`` is either a list of qubits or ``None``
    to apply gate time to this gate one any set of qubits,
    and ``value`` is the gate time in nanoseconds.

    The resulting errors may contains two types of errors: gate errors and relaxation errors.
    The gate errors are generated only for ``Gate`` objects while the relaxation errors are
    generated for all ``Instruction`` objects. Exceptionally, no ``QuantumError`` s are
    generated for ``Measure`` since ``ReadoutError`` s are generated separately instead.

    Args:
        properties (BackendProperties): device backend properties.
        gate_error (bool): Include depolarizing gate errors (Default: True).
        thermal_relaxation (Bool): Include thermal relaxation errors
                                   (Default: True).
        gate_lengths (list): Override device gate times with custom
                             values. If None use gate times from
                             backend properties. (Default: None).
        gate_length_units (str): Time units for gate length values in gate_lengths.
                                 Can be 'ns', 'ms', 'us', or 's' (Default: 'ns').
        temperature (double): qubit temperature in milli-Kelvin (mK)
                              (Default: 0).
        standard_gates (bool): DEPRECATED, If true return errors as standard
                               qobj gates. If false return as unitary
                               qobj instructions (Default: None).
        warnings (bool): PLAN TO BE DEPRECATED, Display warnings (Default: None).
        target (Target): device backend target (Default: None). When this is supplied,
                         several options are disabled:
                         `properties`, `gate_lengths` and `gate_length_units` are not used
                         during the construction of gate errors.
                         Default values are always used for `standard_gates` and `warnings`.

    Returns:
        list: A list of tuples ``(label, qubits, QuantumError)``, for gates
        with non-zero quantum error terms, where `label` is the label of the
        noisy gate, `qubits` is the list of qubits for the gate.

    Raises:
        NoiseError: If invalid arguments are supplied.
    """
    if properties is None and target is None:
        raise NoiseError("Either properties or target must be supplied.")

    if standard_gates is not None:
        warn(
            '"standard_gates" option has been deprecated as of qiskit-aer 0.10.0'
            ' and will be removed no earlier than 3 months from that release date.',
            DeprecationWarning, stacklevel=2)

    if warnings is not None:
        warn(
            '"warnings" argument will be deprecated as part of the qiskit-aer 0.12.0 and '
            'subsequently removed',
            PendingDeprecationWarning, stacklevel=2)
    else:
        warnings = True

    if target is not None:
        if standard_gates is not None or not warnings:
            warn("When `target` is supplied, `standard_gates` and `warnings` are ignored,"
                 " and they are always set to true.", UserWarning)

        if gate_lengths:
            raise NoiseError("When `target` is supplied, `gate_lengths` option is not allowed."
                             "Use `duration` property in target's InstructionProperties instead.")

        return _basic_device_target_gate_errors(
            target=target,
            gate_error=gate_error,
            thermal_relaxation=thermal_relaxation,
            temperature=temperature
        )

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
                # Convert all gate lengths to nanosecond units
                time = value * _NANOSECOND_UNITS[gate_length_units]
                if name in custom_times:
                    custom_times[name].append((qubits, time))
                else:
                    custom_times[name] = [(qubits, time)]
    # Get the device gate parameters from properties
    device_gate_params = gate_param_values(properties)

    # Construct quantum errors
    errors = []
    for name, qubits, gate_length, error_param in device_gate_params:
        # Initilize empty errors
        depol_error = None
        relax_error = None
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
        # Get relaxation error
        if thermal_relaxation:
            relax_error = _device_thermal_relaxation_error(
                qubits, relax_time, relax_params, temperature,
                thermal_relaxation)

        # Get depolarizing error channel
        if gate_error:
            with catch_warnings():
                filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    module="qiskit_aer.noise.errors.errorutils"
                )
                depol_error = _device_depolarizing_error(
                    qubits, error_param, relax_error, standard_gates)

        # Combine errors
        combined_error = _combine_depol_and_relax_error(depol_error, relax_error)
        if combined_error:
            errors.append((name, qubits, combined_error))

    return errors


def _combine_depol_and_relax_error(depol_error, relax_error):
    if depol_error and relax_error:
        return depol_error.compose(relax_error)
    if depol_error:
        return depol_error
    if relax_error:
        return relax_error
    return None


def _basic_device_target_gate_errors(target,
                                     gate_error=True,
                                     thermal_relaxation=True,
                                     temperature=0):
    """Return QuantumErrors derived from a devices Target.
    Note that, in the resulting error list, non-Gate instructions (e.g. Reset) will have
    no gate errors while they may have thermal relaxation errors. Exceptionally,
    Measure instruction will have no errors, neither gate errors nor relaxation errors.
    """
    errors = []
    for op_name, inst_prop_dic in target.items():
        operation = target.operation_from_name(op_name)
        if isinstance(operation, Measure):
            continue
        if inst_prop_dic is None:  # ideal simulator
            continue
        for qubits, inst_prop in inst_prop_dic.items():
            if inst_prop is None:
                continue
            depol_error = None
            relax_error = None
            # Get relaxation error
            if thermal_relaxation and inst_prop.duration:
                relax_params = {q: (target.qubit_properties[q].t1,
                                    target.qubit_properties[q].t2,
                                    target.qubit_properties[q].frequency)
                                for q in qubits}
                relax_error = _device_thermal_relaxation_error(
                    qubits=qubits,
                    gate_time=inst_prop.duration,
                    relax_params=relax_params,
                    temperature=temperature,
                )
            # Get depolarizing error
            if gate_error and inst_prop.error and isinstance(operation, Gate):
                depol_error = _device_depolarizing_error(
                    qubits=qubits,
                    error_param=inst_prop.error,
                    relax_error=relax_error,
                )
            # Combine errors
            combined_error = _combine_depol_and_relax_error(depol_error, relax_error)
            if combined_error:
                errors.append((op_name, qubits, combined_error))

    return errors


def _device_depolarizing_error(qubits,
                               error_param,
                               relax_error=None,
                               standard_gates=True):
    """Construct a depolarizing_error for device.
    If un-physical parameters are supplied, they are truncated to the theoretical bound values."""

    # We now deduce the depolarizing channel error parameter in the
    # presence of T1/T2 thermal relaxation. We assume the gate error
    # parameter is given by e = 1 - F where F is the average gate fidelity,
    # and that this average gate fidelity is for the composition
    # of a T1/T2 thermal relaxation channel and a depolarizing channel.

    # For the n-qubit depolarizing channel E_dep = (1-p) * I + p * D, where
    # I is the identity channel and D is the completely depolarizing
    # channel. To compose the errors we solve for the equation
    # F = F(E_dep * E_relax)
    #   = (1 - p) * F(I * E_relax) + p * F(D * E_relax)
    #   = (1 - p) * F(E_relax) + p * F(D)
    #   = F(E_relax) - p * (dim * F(E_relax) - 1) / dim

    # Hence we have that the depolarizing error probability
    # for the composed depolarization channel is
    # p = dim * (F(E_relax) - F) / (dim * F(E_relax) - 1)
    if relax_error is not None:
        relax_fid = qi.average_gate_fidelity(relax_error)
        relax_infid = 1 - relax_fid
    else:
        relax_fid = 1
        relax_infid = 0
    if error_param is not None and error_param > relax_infid:
        num_qubits = len(qubits)
        dim = 2 ** num_qubits
        error_max = dim / (dim + 1)
        # Check if reported error param is un-physical
        # The minimum average gate fidelity is F_min = 1 / (dim + 1)
        # So the maximum gate error is 1 - F_min = dim / (dim + 1)
        if error_param > error_max:
            error_param = error_max
        # Model gate error entirely as depolarizing error
        num_qubits = len(qubits)
        dim = 2 ** num_qubits
        depol_param = dim * (error_param - relax_infid) / (dim * relax_fid - 1)
        max_param = 4**num_qubits / (4**num_qubits - 1)
        if depol_param > max_param:
            depol_param = min(depol_param, max_param)
        with catch_warnings():
            filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module="qiskit_aer.noise.device.models"
            )
            return depolarizing_error(
                depol_param, num_qubits, standard_gates=standard_gates)
    return None


def _device_thermal_relaxation_error(qubits,
                                     gate_time,
                                     relax_params,
                                     temperature,
                                     thermal_relaxation=True):
    """Construct a thermal_relaxation_error for device"""
    # Check trivial case
    if not thermal_relaxation or gate_time is None or gate_time == 0:
        return None

    # Construct a tensor product of single qubit relaxation errors
    # for any multi qubit gates
    first = True
    error = None
    for qubit in qubits:
        t1, t2, freq = relax_params[qubit]
        t2 = _truncate_t2_value(t1, t2)
        population = _excited_population(freq, temperature)
        if first:
            error = thermal_relaxation_error(t1, t2, gate_time, population)
            first = False
        else:
            single = thermal_relaxation_error(t1, t2, gate_time, population)
            error = error.expand(single)
    return error


def _truncate_t2_value(t1, t2):
    """Return t2 value truncated to 2 * t1 (for t2 > 2 * t1)"""
    new_t2 = t2
    if t2 > 2 * t1:
        new_t2 = 2 * t1
    return new_t2


def _excited_population(freq, temperature):
    """Return excited state population from freq [GHz] and temperature [mK]."""
    population = 0
    if freq != inf and temperature != 0:
        # Compute the excited state population from qubit frequency and temperature
        # based on Maxwell-Boltzmann distribution
        # considering only qubit states (|0> and |1>), i.e. truncating higher energy states.
        # Boltzman constant  kB = 8.617333262e-5 (eV/K)
        # Planck constant h = 4.135667696e-15 (eV.s)
        # qubit temperature temperatue = T (mK)
        # qubit frequency frequency = f (GHz)
        # excited state population = 1/(1+exp((h*f*1e9)/(kb*T*1e-3)))
        # See e.g. Phys. Rev. Lett. 114, 240501 (2015).
        exp_param = exp((47.99243 * freq) / abs(temperature))
        population = 1 / (1 + exp_param)
        if temperature < 0:
            # negative temperate implies |1> is thermal ground
            population = 1 - population
    return population
