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
Instruction Library (:mod:`qiskit_aer.library`)
=========================================================

.. currentmodule:: qiskit_aer.library

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
    SetSuperOp
    SetUnitary
    SetMatrixProductState

QuantumCircuit Methods
----------------------

The set instructions can also be added to circuits by using the
following ``QuantumCircuit`` methods which are patched when importing Aer.

.. autosummary::
    :toctree: ../stubs/

    set_statevector
    set_density_matrix
    set_stabilizer
    set_unitary
    set_superop
    set_matrix_product_state


Saving Simulator Data
=====================

Simulator State Save Instruction Classes
----------------------------------------

The following instructions can be used to save the state of the simulator
into the returned result object. The :class:`SaveState` instruction will
automatically select the format based on the simulation method (eg.
:class:`SaveStatevector` for statevector method, :class:`SaveDensityMatrix`
for density matrix method etc.).

.. autosummary::
    :toctree: ../stubs/

    SaveState
    SaveStatevector
    SaveStatevectorDict
    SaveDensityMatrix
    SaveMatrixProductState
    SaveClifford
    SaveStabilizer
    SaveSuperOp
    SaveUnitary

.. note::
    The :class:`SaveDensityMatrix` instruction can be used to save the
    reduced densit matrix of a subset of qubits for supported simulation
    methods, however all other save state instructions must be applied
    to all qubits in a run circuit.

.. note::
    The :class:`~qiskit_aer.StatevectorSimulator` (and
    :class:`~qiskit_aer.UnitarySimulator`) backend automatically
    append every run circuit with the a :func:`SaveStatevector``
    (:func:`SaveUnitary``) instruction using the default label. Hence adding
    any additional save instructions of that type will require specifying a
    custom label for those instructions.

Simulator Derived Data Save Instruction Classes
-----------------------------------------------

The following classes can be used to directly save data derived from the
simulator state to the returned result object. One some are compatible
with certain simulation methods.

For convenience the save instructions can be accessed using
custom ``QuantumCircuit`` methods

.. autosummary::
    :toctree: ../stubs/

    SaveExpectationValue
    SaveExpectationValueVariance
    SaveProbabilities
    SaveProbabilitiesDict
    SaveAmplitudes
    SaveAmplitudesSquared

.. note ::
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

QuantumCircuit Methods
----------------------

The save instructions can also be added to circuits by using the
following ``QuantumCircuit`` methods which are patched when importing Aer.

.. note ::
    Each save method has a default label for accessing from the
    circuit result data, however duplicate labels in results will result
    in an exception being raised. If you use more than 1 instance of a
    specific save instruction you must set a custom label for the
    additional instructions.

.. autosummary::
    :toctree: ../stubs/

    save_amplitudes
    save_amplitudes_squared
    save_clifford
    save_density_matrix
    save_expectation_value
    save_expectation_value_variance
    save_matrix_product_state
    save_probabilities
    save_probabilities_dict
    save_stabilizer
    save_state
    save_statevector
    save_statevector_dict
    save_superop
    save_unitary

Method Compatibility
====================

The following table summarizes which instructions are compatible with
which simulation methods

.. csv-table::
    :file: instructions_table.csv
    :header-rows: 1
"""

__all__ = [
    "SaveAmplitudes",
    "SaveAmplitudesSquared",
    "SaveClifford",
    "SaveDensityMatrix",
    "SaveExpectationValue",
    "SaveExpectationValueVariance",
    "SaveMatrixProductState",
    "SaveProbabilities",
    "SaveProbabilitiesDict",
    "SaveStabilizer",
    "SaveState",
    "SaveStatevector",
    "SaveStatevectorDict",
    "SaveSuperOp",
    "SaveUnitary",
    "SetDensityMatrix",
    "SetStabilizer",
    "SetStatevector",
    "SetSuperOp",
    "SetUnitary",
    "SetMatrixProductState",
]

from .save_instructions import *
from .set_instructions import *
