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
=============================================
Utilities (:mod:`qiskit_aer.utils`)
=============================================

.. currentmodule:: qiskit_aer.utils

This module contains utility functions for modifying
:class:`~qiskit_aer.noise.NoiseModel` objects and ``QuantumCircuits``
using noise models.


Functions
=========

.. autosummary::
    :toctree: ../stubs/

    insert_noise
    approximate_quantum_error
    approximate_noise_model
    transform_noise_model
    transpile_noise_model
    transpile_quantum_error
"""

from .noise_model_inserter import insert_noise
from .noise_transformation import approximate_noise_model
from .noise_transformation import approximate_quantum_error
from .noise_transformation import transform_noise_model
from .noise_transformation import transpile_noise_model
from .noise_transformation import transpile_quantum_error
