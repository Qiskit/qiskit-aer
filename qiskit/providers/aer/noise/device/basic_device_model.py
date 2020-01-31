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
# pylint: disable=invalid-name,import-outside-toplevel
"""
Simplified noise models for devices backends.
"""

import warnings


def basic_device_noise_model(properties,
                             gate_error=True,
                             readout_error=True,
                             thermal_relaxation=True,
                             temperature=0,
                             gate_lengths=None,
                             standard_gates=True):
    """
    Return a noise model derived from a devices backend properties.

    This function generates a noise model based on:

    * 1 and 2 qubit gate errors consisting of a
      :func:`depolarizing_error` followed
      by a :func:`thermal_relaxation_error`.

    * Single qubit :class:`ReadoutError` on all measurements.

    The Error error parameters are tuned for each individual qubit based on
    the :math:`T_1`, :math:`T_2`, frequency and readout error parameters for
    each qubit, and the gate error and gate time parameters for each gate
    obtained from the device backend properties.

    **Additional Information**

    The noise model includes the following errors:

    * If ``readout_error=True`` include single qubit readout
      errors on measurements.

    * If ``gate_error=True`` and ``thermal_relaxation=True`` include:

        * Single-qubit gate errors consisting of a :func:`depolarizing_error`
          followed by a :func:`thermal_relaxation_error` for the qubit the
          gate acts on.

        * Two-qubit gate errors consisting of a 2-qubit
          :func:`depolarizing_error` followed by single qubit
          :func:`thermal_relaxation_error` on each qubit participating in
          the gate.

    * If ``gate_error=True`` is ``True`` and ``thermal_relaxation=False``:

        * An N-qubit :func:`depolarizing_error` on each N-qubit gate.

    * If ``gate_error=False`` and ``thermal_relaxation=True`` include
      single-qubit :func:`thermal_relaxation_errors` on each qubits
      participating in a multi-qubit gate.

    For best practice in simulating a backend make sure that the
    circuit is compiled using the set of basis gates in the noise
    module by setting ``basis_gates=noise_model.basis_gates``
    and using the device coupling map with
    ``coupling_map=backend.configuration().coupling_map``

    **Specifying custom gate times**

    The ``gate_lengths`` kwarg can be used to specify custom gate times
    to add gate errors using the :math:`T_1` and :math:`T_2` values from
    the backend properties. This should be passed as a list of tuples
    ``gate_lengths=[(name, value), ...]``
    where ``name`` is the gate name string, and ``value`` is the gate time
    in nanoseconds.

    If a custom gate is specified that already exists in
    the backend properties, the ``gate_lengths`` value will override the
    gate time value from the backend properties.
    If non-default values are used gate_lengths should be a list

    Args:
        properties (BackendProperties): backend properties.
        gate_error (bool): Include depolarizing gate errors (Default: True).
        readout_error (Bool): Include readout errors in model
                              (Default: True).
        thermal_relaxation (Bool): Include thermal relaxation errors
                                   (Default: True).
        temperature (double): qubit temperature in milli-Kelvin (mK) for
                              thermal relaxation errors (Default: 0).
        gate_lengths (list): Custom gate times for thermal relaxation errors.
                             Used to extend or override the gate times in
                             the backend properties (Default: None))
        standard_gates (bool): If true return errors as standard
                               qobj gates. If false return as unitary
                               qobj instructions (Default: True)

    Returns:
        NoiseModel: An approximate noise model for the device backend.
    """
    warnings.warn(
        'This function is been deprecated and moved to a method of the'
        '`NoiseModel` class. For equivalent functionality use'
        ' `NoiseModel.from_backend(properties, **kwargs).',
        DeprecationWarning)
    # This wrapper is for the deprecated function
    # We need to import noise model here to avoid cyclic import errors
    # pylint: disable=import-outside-toplevel
    from qiskit.providers.aer.noise.noise_model import NoiseModel
    return NoiseModel.from_backend(properties,
                                   gate_error=gate_error,
                                   readout_error=readout_error,
                                   thermal_relaxation=thermal_relaxation,
                                   temperature=temperature,
                                   gate_lengths=gate_lengths,
                                   standard_gates=standard_gates)
