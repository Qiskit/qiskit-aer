# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===================================================
Primitives (:mod:`qiskit_aer.primitives`)
===================================================

.. currentmodule:: qiskit_aer.primitives

This module is Aer implementation of primitives.
See the docs https://docs.quantum.ibm.com/api/qiskit/primitives for general descriptions.


Classes
=======

.. autosummary::
    :toctree: ../stubs/

    Sampler
    Estimator
"""

from .estimator import Estimator
from .estimator_v2 import EstimatorV2
from .sampler import Sampler
from .sampler_v2 import SamplerV2
