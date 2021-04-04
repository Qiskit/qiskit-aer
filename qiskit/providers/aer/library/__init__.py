# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
=========================================================
Instruction Library (:mod:`qiskit.providers.aer.library`)
=========================================================

.. currentmodule:: qiskit.providers.aer.library

This library contains custom qiskit :class:`~qiskit.QuantumCircuit`
:class:`~qiskit.circuit.Instruction` subclasses that can be used
with the Aer circuit simulator backends.

Setting a Custom Simulator State
================================

The following instruction classes can be used to set the specific
simulator methods to a custom state. Note that these instructions
are only valid when applied to all qubits in a circuit. Applying
to a subset of qubits will raise an exception during execution.

Instruction Classes
-------------------

.. autosummary::
    :toctree: ../stubs/

    SetStatevector
    SetDensityMatrix
    SetStabilizer
    SetUnitary
    SetMatrixProductState

Then can also be used using custom QuantumCircuit methods

QuantumCircuit Methods
----------------------

.. autosummary::
    :toctree: ../stubs/

    set_statevector
    set_density_matrix
    set_stabilizer
    set_unitary
    set_matrix_product_state


Saving Simulator Data
=====================

The following classes can be used to directly save data from the
simulator to the returned result object.

Instruction Classes
-------------------

.. autosummary::
    :toctree: ../stubs/

    SaveState
    SaveExpectationValue
    SaveExpectationValueVariance
    SaveProbabilities
    SaveProbabilitiesDict
    SaveDensityMatrix
    SaveStatevector
    SaveStatevectorDict
    SaveAmplitudes
    SaveAmplitudesSquared
    SaveStabilizer
    SaveUnitary
    SaveMatrixProductState

Then can also be used using custom QuantumCircuit methods

QuantumCircuit Methods
----------------------

.. autosummary::
    :toctree: ../stubs/

    save_state
    save_expectation_value
    save_expectation_value_variance
    save_probabilities
    save_probabilities_dict
    save_density_matrix
    save_statevector
    save_statevector_dict
    save_amplitudes
    save_amplitudes_squared
    save_stabilizer
    save_unitary
    save_matrix_product_state

.. note ::

    **Save Instruction Labels**

    Each save instruction has a default label for accessing from the
    circuit result data, however duplicate labels in results will result
    in an exception being raised. If you use more than 1 instance of a
    specific save instruction you must set a custom label for the
    additional instructions.

.. note ::

    **Pershot Data with Measurement Sampling Optimization**

    When saving pershot data by using the ``pershot=True`` kwarg
    in the above instructions, the resulting list may only contain
    a single value rather than the number of shots. This
    happens when a run circuit supports measurement sampling because
    it is either

    1. An ideal simulation with all measurements at the end.

    2. A noisy simulation using the density matrix method with all
    measurements at the end.

    In both these cases only a single shot is actually simulated and
    measurement samples for all shots are calculated from the final
    state.
"""

__all__ = [
    'SaveState',
    'SaveExpectationValue',
    'SaveExpectationValueVariance',
    'SaveProbabilities',
    'SaveProbabilitiesDict',
    'SaveStatevector',
    'SaveStatevectorDict',
    'SaveDensityMatrix',
    'SaveAmplitudes',
    'SaveAmplitudesSquared',
    'SaveUnitary',
    'SaveStabilizer',
    'SaveMatrixProductState'
]

from .save_instructions import *
from .set_instructions import *
