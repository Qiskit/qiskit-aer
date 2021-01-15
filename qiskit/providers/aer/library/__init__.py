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

Saving Simulator Data
=====================

The following classes can be used to directly save data from the
simulator to the returned result object.

Instruction Classes
-------------------

.. autosummary::
    :toctree: ../stubs/

    SaveExpval

Then can also be used using custom QuantumCircuit methods

QuantumCircuit Methods
----------------------

.. autosummary::
    :toctree: ../stubs/

    save_expval
"""

__all__ = ['SaveExpval']

from .save_expval import SaveExpval, save_expval
