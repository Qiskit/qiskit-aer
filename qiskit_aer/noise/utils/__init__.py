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
"""
The functions here have been moved to `qiskit.providers.aer.utils`.
"""

import warnings as warn

# DEPRECATED: these functions have been moved
from ...utils import remap_noise_model as _remap_noise_model
from ...utils import NoiseTransformer as _NoiseTransformer
from ...utils import approximate_quantum_error as _approximate_quantum_error
from ...utils import approximate_noise_model as _approximate_noise_model
from ...utils import insert_noise as _insert_noise


def remap_noise_model(noise_model,
                      remapping,
                      discard_qubits=False,
                      warnings=True):
    """Remap qubits in a noise model.

    This remaps the specified gate qubits for local quantum errors, the gate
    and noise qubits for non-local quantum errors, and the gate qubits for
    local ReadoutErrors. All-qubit quantum and readout errors are unaffected.

    Args:
        noise_model (NoiseModel): a noise model to remap qubits.
        remapping (list): list or remappings of old qubit to new qubit.
                          See Additional Information.
        discard_qubits (bool): if True discard qubits not in remapping keys,
                               if False an identity mapping wil be assumed
                               for unnamed qubits (Default: False).
        warnings (bool): display warnings if qubits being remapped are not
                         in the input noise model (Default: True).

    Returns:
        NoiseModel: a new noise model with the same errors but remapped
        gate and noise qubits for local and non-local errors.

    Raises:
        NoiseError: if remapping has duplicate qubits in the remapped qubits.

    Additional Information:
        * The remapping map be specified as either a list of pairs:
          ``[(old, new), ...]``, or a list of old qubits where the new qubit is
          inferred from the position: ``[old0, old1, ...]`` is treated as
          ``[(old0, 0), (old1, 1), ...]``.

        * If ``discard_qubits`` is ``False``, any qubits in the noise model not
          specified in the list of old qubits will be added to the remapping as
          a trivial mapping ``(qubit, qubit)``.
    """
    warn.warn(
        'This function is been moved to `qiskit.providers.aer.utils.remap_noise_model`.'
        ' Importing it from `qiskit.providers.aer.noise.utils` will be'
        ' removed in a future release.', DeprecationWarning)
    return _remap_noise_model(noise_model,
                              remapping,
                              discard_qubits=discard_qubits,
                              warnings=warnings)


def insert_noise(circuits, noise_model, transpile=False):
    """Return a noisy version of a QuantumCircuit.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): Input noise-free circuits.
        noise_model (NoiseModel):  The noise model containing the errors to add
        transpile (Boolean): Should the circuit be transpiled into the noise model basis gates

    Returns:
        QuantumCircuit: The new circuit with the Kraus noise instructions inserted.

    Additional Information:
        The noisy circuit return by this function will consist of the
        original circuit with ``Kraus`` instructions inserted after all
        instructions referenced in the ``noise_model``. The resulting circuit
        cannot be ran on a quantum computer but can be executed on the
        :class:`~qiskit.providers.aer.QasmSimulator`.
    """
    warn.warn(
        'This function is been moved to `qiskit.providers.aer.utils.insert_noise`.'
        ' Importing it from `qiskit.providers.aer.noise.utils` will be'
        ' removed in a future release.', DeprecationWarning)
    return _insert_noise(circuits, noise_model, transpile=transpile)


def approximate_quantum_error(error,
                              *,
                              operator_string=None,
                              operator_dict=None,
                              operator_list=None):
    """
    Return an approximate QuantumError bases on the Hilbert-Schmidt metric.

    Currently this is only implemented for 1-qubit QuantumErrors.

    Args:
        error (QuantumError): the error to be approximated.
        operator_string (string or None): a name for a pre-made set of
            building blocks for the output channel (Default: None).
        operator_dict (dict or None): a dictionary whose values are the
            building blocks for the output channel (Default: None).
        operator_list (dict or None): list of building blocks for the
            output channel (Default: None).

    Returns:
        QuantumError: the approximate quantum error.

    Raises:
        NoiseError: if number of qubits is not supported or approximation
                    failed.
        RuntimeError: If there's no information about the noise type.

    Additional Information:
        The operator input precedence is: ``list`` < ``dict`` < ``str``.
        If a string is given, dict is overwritten; if a dict is given, list is
        overwritten. Oossible values for string are ``'pauli'``, ``'reset'``,
        ``'clifford'``.
        For further information see :meth:`NoiseTransformer.named_operators`.
    """
    warn.warn(
        'This function is been moved to `qiskit.providers.aer.utils.approximate_qauntum_error`.'
        ' Importing it from `qiskit.providers.aer.noise.utils` will be removed'
        ' in a future release.',
        DeprecationWarning)
    return _approximate_quantum_error(error,
                                      operator_string=operator_string,
                                      operator_dict=operator_dict,
                                      operator_list=operator_list)


def approximate_noise_model(model,
                            *,
                            operator_string=None,
                            operator_dict=None,
                            operator_list=None):
    """
    Return an approximate noise model.

    Args:
        model (NoiseModel): the noise model to be approximated.
        operator_string (string or None): a name for a pre-made set of
            building blocks for the output channel (Default: None).
        operator_dict (dict or None): a dictionary whose values are the
            building blocks for the output channel (Default: None).
        operator_list (dict or None): list of building blocks for the
            output channel (Default: None).

    Returns:
        NoiseModel: the approximate noise model.

    Raises:
        NoiseError: if number of qubits is not supported or approximation
        failed.

    Additional Information:
        The operator input precedence is: ``list`` < ``dict`` < ``str``.
        If a string is given, dict is overwritten; if a dict is given, list is
        overwritten. Oossible values for string are ``'pauli'``, ``'reset'``,
        ``'clifford'``.
        For further information see :meth:`NoiseTransformer.named_operators`.
    """
    warn.warn(
        'This function is been moved to `qiskit.providers.aer.utils.approximate_noise_model`.'
        ' Importing it from `qiskit.providers.aer.noise.utils` will be removed in a'
        ' future release.', DeprecationWarning)
    return _approximate_noise_model(model,
                                    operator_string=operator_string,
                                    operator_dict=operator_dict,
                                    operator_list=operator_list)


class NoiseTransformer(_NoiseTransformer):
    """Transforms one quantum channel to another based on a specified criteria."""
    def __init__(self):
        warn.warn(
            'This function is been moved to `qiskit.providers.aer.utils.NoiseTransformer`.'
            ' Importing it from `qiskit.providers.aer.noise.utils` will be removed in a'
            ' future release.', DeprecationWarning)
        super().__init__()
